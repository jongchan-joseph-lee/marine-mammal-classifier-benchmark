"""Extend compatible frozen embeddings without recomputing original core rows.

AST zero/tile and legacy pre-activation VGGish are matched to the original
reference arrays by real-audio numerical checks before accepting reuse.
CUDA is optional; precision stays float32 and TF32 is disabled.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
sys.path.insert(0,str(BASE / 'deep_review/raw_new/src'))
from b06_embed_pretrained import AstEmbedder, VggishEmbedder, prepare, tile_to
from wb_common import read_audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='vggish,ast_zero,ast_tile')
    ap.add_argument('--device', choices=['auto','cpu','cuda'], default='auto')
    ap.add_argument('--threads', type=int, default=6)
    ap.add_argument('--batch-size', type=int, default=8)
    ap.add_argument('--model-revision', default=None,
                    help='Optional immutable Hugging Face AST model revision.')
    ap.add_argument('--calibration-only', action='store_true')
    ap.add_argument('--audio-wait-timeout', type=float, default=60.,
                    help='Stop if no missing-cut audio becomes available within this many seconds.')
    args = ap.parse_args()
    device = 'cuda' if args.device=='auto' and torch.cuda.is_available() else ('cpu' if args.device=='auto' else args.device)
    if device=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
    torch.set_num_threads(args.threads)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
    m = pd.read_csv(BASE / 'deep_review/raw_new/data/bench_manifest.csv', dtype={'cut_id':str}, low_memory=False).fillna('')
    m = m[m.pool.eq('main') & m.hold_reason.eq('')].sort_values('cut_id').reset_index(drop=True)
    assert len(m)==10694
    out = ROOT / 'output/features'; out.mkdir(parents=True,exist_ok=True)
    prov = ROOT / 'output/provenance'; prov.mkdir(parents=True,exist_ok=True)
    ast=None
    for feature in args.features.split(','):
        source = BASE / f'deep_review/raw/features/{feature}.npz'
        z=np.load(source,allow_pickle=False)
        frozen_info=json.loads(str(z['info'].item()))
        names=z['names'].astype(str)
        # Completed arrays are reusable too, including on a later CUDA host.
        # Keep the frozen source hash and representation semantics as guards.
        target=out/(feature+'.npz')
        if target.exists() and not args.calibration_only:
            complete=np.load(target,allow_pickle=False)
            info=json.loads(str(complete['info'].item()));saved=info.get('settings',{})
            compatible=(info.get('full_coverage',False)
                and saved.get('original_array_sha256')==hashlib.sha256(source.read_bytes()).hexdigest()
                and saved.get('precision')=='fp32' and saved.get('max_analysis_seconds')==120
                and saved.get('vggish_output_pre_activation')==(feature=='vggish')
                and np.array_equal(complete['cut_id'].astype(str),m.cut_id.to_numpy(dtype=str))
                and complete['X'].shape==(len(m),len(names)) and np.isfinite(complete['X']).all()
                and np.array_equal(complete['names'].astype(str),names))
            if compatible:
                print(f'{feature}: compatible completed full-cohort array reused ({len(m)} cuts).',flush=True)
                continue
        frozen={str(c):x for c,x in zip(z['cut_id'],z['X'])}
        values={c:frozen[c] for c in m.cut_id if c in frozen}
        assert len(values)==3539
        if feature.startswith('ast_'):
            if ast is None: ast=AstEmbedder(feature.split('_',1)[1], revision=args.model_revision)
            ast.pad=feature.split('_',1)[1]
            emb=ast
        elif feature=='vggish':
            emb=VggishEmbedder()
            # The original vggish-10086976 arrays were saved before the final
            # ReLU. This is a distinct representation from the later official
            # post-ReLU run; preserve the original representation exactly.
            emb.model.embedding_network[-1]=torch.nn.Identity()
        else: raise ValueError(feature)
        emb.model.to(device).eval()
        torch.set_num_threads(args.threads)
        loaded_revision=getattr(getattr(emb.model,'config',None),'_commit_hash',None)
        if feature.startswith('ast_') and loaded_revision is None:
            from huggingface_hub import try_to_load_from_cache
            config_path=try_to_load_from_cache(frozen_info['cache_settings']['model_id'],
                                               'config.json',revision=args.model_revision)
            if isinstance(config_path,str):
                resolved=Path(config_path).parent.name
                if len(resolved)==40 and all(c in '0123456789abcdef' for c in resolved):
                    loaded_revision=resolved
        settings=dict(feature=feature,device=device,precision='fp32',tf32=False,threads=args.threads,
                      model_id=frozen_info['cache_settings']['model_id'],max_analysis_seconds=120,
                      original_array_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                      original_array_info=frozen_info,torch=torch.__version__,adapter_version=1,
                      vggish_output_pre_activation=(feature=='vggish'),
                      loaded_revision=loaded_revision)
        semantic=json.dumps({k:v for k,v in settings.items() if k not in ['device','threads']},sort_keys=True)
        token=hashlib.sha256(semantic.encode()).hexdigest()
        cache=ROOT / 'work/embedding_cache' / feature; cache.mkdir(parents=True,exist_ok=True)

        def batch(waves):
            chunks=[];bounds=[0]
            for w in waves:
                if feature=='vggish':
                    x=tile_to(w,16000).astype(np.float32)
                    chunks.append(emb.proc(torch.from_numpy(x)))
                    bounds.append(bounds[-1]+len(chunks[-1]))
                else:
                    chunks.extend(emb.chunks(w));bounds.append(len(chunks))
            with torch.inference_mode():
                if feature=='vggish':
                    inputs=torch.cat(chunks,dim=0)
                    output=torch.cat([emb.model(inputs[i:i+48].to(device)).float().cpu()
                                      for i in range(0,len(inputs),48)]).numpy()
                else:
                    inputs=emb.fe([c.astype(np.float32) for c in chunks],sampling_rate=16000,return_tensors='pt')
                    count=inputs['input_values'].shape[0]
                    output=np.concatenate([emb.model(**{k:v[i:i+args.batch_size].to(device) for k,v in inputs.items()}).pooler_output.float().cpu().numpy()
                                           for i in range(0,count,args.batch_size)])
            return np.stack([output[bounds[i]:bounds[i+1]].mean(0) for i in range(len(waves))])

        def wave(cid):
            x,fs=read_audio(str(ROOT / 'audio_original' / (cid+'.flac')),120.)
            return prepare(x,fs,False)

        # Preserve an auditable numerical compatibility threshold. Calibration
        # is on embeddings only, not performance or research-label agreement.
        available=m[m.cut_id.isin(values)].copy()
        ready=available[available.cut_id.map(lambda c:(ROOT/'audio_original'/(c+'.flac')).exists())]
        if len(ready)<12: raise RuntimeError('Need at least 12 real core files for compatibility checks')
        # Include shorter and longer clips when available to test patch means.
        ordered=ready.sort_values('dur_s')
        calib=ordered.iloc[np.unique(np.linspace(0,len(ordered)-1,12).astype(int))].cut_id.tolist()
        checks=[];started=time.time()
        for i in range(0,len(calib),args.batch_size):
            part=calib[i:i+args.batch_size]
            result=batch([wave(c) for c in part])
            for c,a in zip(part,result):
                b=frozen[c]
                cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))
                diff=float(np.max(abs(a-b)))
                ok=bool(np.allclose(a,b,rtol=3e-4,atol=3e-4) and cosine>.999999)
                checks.append(dict(cut_id=c,feature=feature,max_abs=diff,cosine=cosine,passed=ok))
        (prov / (feature+'_cache_equivalence.json')).write_text(json.dumps(checks,indent=2))
        if not all(r['passed'] for r in checks):
            raise RuntimeError(f'{feature} is incompatible with frozen arrays: {checks}')
        calibration_seconds=time.time()-started
        print(f'{feature}: cache compatibility passed on {len(calib)} cuts; {calibration_seconds:.1f}s; {device}; 3539 rows reused.',flush=True)
        if args.calibration_only: continue
        remaining={r.cut_id:r for r in m.itertuples(index=False) if r.cut_id not in values}
        computed=resumed=0;inference_seconds=0.;last_print=0
        start=time.time();audio_wait_started=None
        while remaining:
            ready=[c for c in remaining if (ROOT/'audio_original'/(c+'.flac')).exists()][:args.batch_size]
            if not ready:
                if audio_wait_started is None: audio_wait_started=time.time()
                status=prov/'audio_validation.json'
                if status.exists() and json.loads(status.read_text())['failed']:
                    raise RuntimeError('Missing original files after downloader failed; refusing a partial result')
                if time.time()-audio_wait_started>=args.audio_wait_timeout:
                    raise RuntimeError(f'No new original audio available for {args.audio_wait_timeout}s; '
                                       f'{len(remaining)} cuts remain. Download originals and resume; caches are preserved.')
                time.sleep(2);continue
            audio_wait_started=None
            new=[];tokens=[]
            for c in ready:
                ck=cache/(c+'.npz');cut_token=token+remaining[c].md5_b64
                if ck.exists():
                    try:
                        zz=np.load(ck,allow_pickle=False)
                        if str(zz['token'].item())==cut_token:
                            vv=zz['X']
                            if vv.shape==(len(names),) and np.isfinite(vv).all():
                                values[c]=vv;resumed+=1;del remaining[c];continue
                    except (OSError,KeyError,ValueError): pass
                new.append(c);tokens.append(cut_token)
            if new:
                waves=[wave(c) for c in new]
                t=time.time();result=batch(waves);inference_seconds+=time.time()-t
                assert result.shape==(len(new),len(names)) and np.isfinite(result).all()
                for c,ct,v in zip(new,tokens,result):
                    v=v.astype(np.float32);values[c]=v
                    tmp=cache/(c+'.npz.pending')
                    with tmp.open('wb') as stream:np.savez_compressed(stream,token=ct,X=v)
                    tmp.replace(cache/(c+'.npz'));computed+=1;del remaining[c]
            if len(values)-last_print>=80 or not remaining:
                elapsed=time.time()-start
                progress=dict(expected=len(m),ready_rows=len(values),remaining=len(remaining),reused_core=3539,
                              computed_new=computed,resumed_new=resumed,elapsed_seconds=elapsed,inference_seconds=inference_seconds,
                              approximate_remaining_seconds=(inference_seconds/computed*len(remaining) if computed else None),device=device)
                (prov/(feature+'_progress.json')).write_text(json.dumps(progress,indent=2))
                print(f'{feature}: {len(values)}/{len(m)}; new {computed}; checkpoint reuse {resumed}; {elapsed:.1f}s',flush=True)
                last_print=len(values)
        ids=m.cut_id.to_numpy(dtype=str);X=np.stack([values[c] for c in ids])
        info=dict(settings=settings,reused_core=3539,computed_new=computed,resumed_new=resumed,
                  seconds=time.time()-start,calibration_seconds=calibration_seconds,expected=len(m),full_coverage=True)
        tmp=out/(feature+'.npz.pending')
        with tmp.open('wb') as stream:np.savez_compressed(stream,cut_id=ids,X=X,names=names,info=json.dumps(info))
        tmp.replace(out/(feature+'.npz'))
        progress.update(complete=True)
        (prov/(feature+'_progress.json')).write_text(json.dumps(progress,indent=2))
        print(f'{feature}: complete full-cohort array saved.',flush=True)
        if feature=='vggish':
            del emb
    print('Compatibility calibration complete.' if args.calibration_only else 'Requested full-cohort embeddings complete.',flush=True)


if __name__=='__main__':main()
