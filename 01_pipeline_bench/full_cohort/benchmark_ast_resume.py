"""Benchmark execution settings only; compare FP32 vectors before any change."""
from pathlib import Path
import json
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT.parent
sys.path.insert(0,str(BASE/'deep_review/raw_new/src'))
from b06_embed_pretrained import AstEmbedder,prepare
from wb_common import read_audio

def main():
    revision='f826b80d28226b62986cc218e5cec390b1096902'
    m=pd.read_csv(ROOT/'output/provenance/main_full_metadata.csv',low_memory=False)
    old=np.load(BASE/'deep_review/raw/features/ast_zero.npz',allow_pickle=False)
    frozen={str(c):x for c,x in zip(old['cut_id'],old['X'])}
    q=m[m.cut_id.isin(frozen)&m.dur_s.between(.5,8)].sort_values('dur_s')
    ids=q.iloc[np.linspace(0,len(q)-1,32).astype(int)].cut_id.tolist()
    waves=[]
    for c in ids:
        x,fs=read_audio(str(ROOT/'audio_original'/(c+'.flac')),120.)
        waves.append(prepare(x,fs,False))
    emb=AstEmbedder('zero',revision=revision)
    torch.set_num_threads(6)
    inputs=emb.fe(waves,sampling_rate=16000,return_tensors='pt')
    results=[]
    for threads,batch in [(6,8),(8,8),(6,16),(8,16),(8,32)]:
        torch.set_num_threads(threads)
        with torch.inference_mode():
            emb.model(**{k:v[:min(batch,8)] for k,v in inputs.items()})
            start=time.perf_counter()
            z=np.concatenate([emb.model(**{k:v[i:i+batch] for k,v in inputs.items()}).pooler_output.float().numpy()
                              for i in range(0,len(ids),batch)])
        elapsed=time.perf_counter()-start
        ref=np.stack([frozen[c] for c in ids])
        ok=bool(np.allclose(z,ref,rtol=3e-4,atol=3e-4))
        row=dict(threads=threads,batch=batch,n=len(ids),seconds=elapsed,
                 cuts_per_second=len(ids)/elapsed,max_abs=float(abs(z-ref).max()),passed=ok)
        results.append(row);print(json.dumps(row),flush=True)
        if not ok:raise RuntimeError('Execution setting changed frozen representation')
    (ROOT/'output/provenance/ast_resume_cpu_benchmark.json').write_text(json.dumps(results,indent=2))

if __name__=='__main__':main()
