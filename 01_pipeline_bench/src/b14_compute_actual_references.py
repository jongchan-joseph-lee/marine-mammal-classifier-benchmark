"""Actual frozen reference inference in batches; strict coverage and resumable caches."""
from __future__ import annotations
import argparse, contextlib, hashlib, importlib.metadata, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from b06_embed_pretrained import (AstEmbedder, VggishEmbedder, AST_NAME, prepare,
                                 cache_token, load_checkpoint, save_checkpoint)
from wb_common import DATA_DIR, load_config, log, read_audio, work_dir

def ast_batch(emb, waves, precision='fp32'):
    torch=emb.torch
    chunks=[];bounds=[0]
    for wave in waves:
        chunks.extend(emb.chunks(wave));bounds.append(len(chunks))
    inputs=emb.fe([x.astype(np.float32) for x in chunks],sampling_rate=16000,return_tensors='pt')
    context=(torch.autocast('cpu',dtype=torch.bfloat16) if precision=='bf16' else contextlib.nullcontext())
    with torch.inference_mode(), context:
        output=emb.model(**inputs).pooler_output.float().cpu().numpy()
    return np.stack([output[bounds[i]:bounds[i+1]].mean(0) for i in range(len(waves))])

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=['ast','vggish'],required=True)
    ap.add_argument('--precision',choices=['fp32','bf16'],default='fp32')
    ap.add_argument('--batch-size',type=int,default=4)
    ap.add_argument('--pad',choices=['zero','wavzero','tile'],default='zero')
    ap.add_argument('--band4k',action='store_true')
    ap.add_argument('--head-seconds',type=float,default=0)
    ap.add_argument('--split',choices=['all','dev','test'],default='all')
    ap.add_argument('--wait-audio',type=int,default=0,help='Wait for a simultaneous downloader, in seconds.')
    ap.add_argument('--limit',type=int,default=0)
    args=ap.parse_args()
    if args.precision=='bf16' and args.model!='ast':ap.error('bf16 is validated for AST only')
    manifest=pd.read_csv(DATA_DIR/'bench_manifest.csv',dtype=str,keep_default_na=False)
    selected=manifest[(manifest.role=='core') & (manifest.pool=='main')]
    if args.split!='all':selected=selected[selected.split==args.split]
    if args.band4k:selected=selected[selected.fs_hz.astype(int)>=8000]
    if args.limit:selected=selected.head(args.limit)
    emb=AstEmbedder(args.pad) if args.model=='ast' else VggishEmbedder()
    name=('ast_'+args.pad if args.model=='ast' else 'vggish')
    if args.head_seconds:name+='_head'+str(args.head_seconds).replace('.','p')
    if args.band4k:name+='_band4k'
    if args.split!='all':name+='_'+args.split
    source_root=Path(load_config()['audio_root'])
    weights_sha=None
    if args.model=='vggish':
        weight_file=Path(emb.torch.hub.get_dir())/'checkpoints/vggish.pt'
        weights_sha=hashlib.sha256(weight_file.read_bytes()).hexdigest()
    settings=dict(model_id=AST_NAME if args.model=='ast' else 'torchaudio/models/vggish.pt',
                  revision=getattr(getattr(emb.model,'config',None),'_commit_hash',None),
                  vggish_weights_sha256=weights_sha,pad=args.pad if args.model=='ast' else 'tile_if_lt_1s',
                  precision=args.precision,band4k=args.band4k,head_seconds=args.head_seconds,
                  torch=emb.torch.__version__,transformers=importlib.metadata.version('transformers'),
                  backend=getattr(emb,'kind','transformers_sdpa'),threads=emb.threads,cache_version=2)
    cache=work_dir('embedding_cache',name);start=time.time()
    remaining={row.cut_id:row for row in selected.itertuples(index=False)}
    values={};metadata=[];resumed=0;last_print=0
    while remaining:
        ready=[cid for cid in remaining if (source_root/(cid+'.flac')).exists()][:args.batch_size]
        if not ready:
            if time.time()-start>args.wait_audio:
                raise RuntimeError(f'{len(remaining)} original audio files missing; refusing a partial result')
            time.sleep(1);continue
        waves=[];new_ids=[];tokens=[];paths=[]
        for cid in ready:
            row=remaining[cid];path=source_root/(cid+'.flac')
            token=cache_token(path,settings);cached=load_checkpoint(cache/(cid+'.npz'),token,emb.dim)
            if cached is not None:values[cid]=cached;resumed+=1
            else:
                x,fs=read_audio(str(path),120.)
                wave=prepare(x,fs,args.band4k)
                if args.head_seconds:wave=wave[:int(round(args.head_seconds*16000))]
                waves.append(wave);new_ids.append(cid);tokens.append(token);paths.append(path)
            del remaining[cid]
        if waves:
            batch_start=time.time()
            batch=ast_batch(emb,waves,args.precision) if args.model=='ast' else np.stack([emb(w) for w in waves])
            if batch.shape!=(len(waves),emb.dim) or not np.isfinite(batch).all():
                raise RuntimeError('Invalid actual-model output; refusing to save')
            for cid,token,vector in zip(new_ids,tokens,batch):
                vector=vector.astype(np.float32);save_checkpoint(cache/(cid+'.npz'),token,vector);values[cid]=vector
            metadata.append(dict(n=len(waves),seconds=time.time()-batch_start))
        if len(values)-last_print>=40 or not remaining:
            log(f'{name}: {len(values)}/{len(selected)} actual embeddings; elapsed {time.time()-start:.0f}s; reused {resumed}')
            last_print=len(values)
    ids=selected.cut_id.to_numpy(dtype=str);X=np.stack([values[cid] for cid in ids])
    info=dict(settings=settings,n_expected=len(selected),n_computed=len(values)-resumed,n_resumed=resumed,
              n_skipped=0,seconds=time.time()-start,batches=metadata,loading_info=getattr(emb,'loading_info',None),
              note='Actual pretrained inference; native FLACs retained; weak note labels are not human gold.')
    target=work_dir('features')/(name+'.npz');temp=target.with_suffix('.npz.tmp')
    with temp.open('wb') as f:np.savez_compressed(f,cut_id=ids,X=X,names=np.array([f'e{i}' for i in range(emb.dim)]),info=json.dumps(info))
    temp.replace(target)
    (work_dir('actual_reference_run')/(name+'_inference.json')).write_text(json.dumps(info,indent=2),encoding='utf-8')
    log(f'Saved {name}: {X.shape}; no skipped files');return 0

if __name__=='__main__':raise SystemExit(main())
