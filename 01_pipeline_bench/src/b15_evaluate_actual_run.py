"""Finish matched comparisons after full actual inference; never evaluate a partial set."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from wb_common import DATA_DIR, PKG_ROOT, work_dir, log

def complete(name, expected=3539):
    p=work_dir('features')/(name+'.npz')
    if not p.exists():return False
    with np.load(p,allow_pickle=False) as z:
        X=z['X']
        if len(X)!=expected:return False
        if not np.isfinite(X).all():raise ValueError('Non-finite embedding: '+name)
        try:
            ids=z['cut_id']
            if len(ids)!=expected:raise ValueError('Cut ID length mismatch: '+name)
            return True
        except ValueError as error:
            if 'Object arrays cannot be loaded' not in str(error):raise
            # Repair only this run's known serialization error, without enabling
            # pickle. The old runner stacked rows in unchanged manifest order.
            metadata=json.loads(z['info'].item())
            if metadata.get('n_expected')!=expected:raise
            m=pd.read_csv(DATA_DIR/'bench_manifest.csv',dtype=str,keep_default_na=False)
            ids=m[(m.role=='core')&(m.pool=='main')].cut_id.to_numpy(dtype=str)
            recorded=json.loads((work_dir('actual_reference_run')/'runtime_and_input_hashes.json').read_text())
            current=hashlib.sha256((DATA_DIR/'bench_manifest.csv').read_bytes()).hexdigest()
            if current!=recorded['input_sha256']['data/bench_manifest.csv'] or len(ids)!=expected:
                raise ValueError('Cannot reconstruct IDs from a changed manifest')
            arrays={k:z[k] for k in z.files if k not in ('cut_id','info')}
            metadata['id_storage_repaired_from_unchanged_manifest']=True
    temporary=p.with_suffix('.npz.tmp')
    with temporary.open('wb') as output:
        np.savez_compressed(output,cut_id=ids,info=json.dumps(metadata),**arrays)
    temporary.replace(p)
    log('Repaired cut ID string dtype; numeric embeddings unchanged: '+name)
    return True

def run(command, filename):
    log('Running '+' '.join(command))
    with (work_dir('actual_reference_run')/filename).open('w',encoding='utf-8') as output:
        subprocess.run([sys.executable,*command],stdout=output,stderr=subprocess.STDOUT,check=True)

def main():
    t0=time.time()
    while not complete('vggish'):
        if time.time()-t0>5400:raise TimeoutError('VGGish full extraction incomplete')
        time.sleep(3)
    os.environ['WATKINS_TORCH_THREADS']='2'
    run([str(PKG_ROOT/'src/b14_compute_actual_references.py'),'--model','vggish','--head-seconds','10.24',
         '--batch-size','8','--wait-audio','1800'],'vggish_head10p24.log')
    while not complete('ast_zero'):
        if time.time()-t0>5400:raise TimeoutError('AST full extraction incomplete')
        time.sleep(3)
    features=['meta','mfcc16k','physv0','vggish','ast_zero','vggish_head10p24']
    core=pd.read_csv(DATA_DIR/'bench_core.csv',dtype=str,keep_default_na=False)
    main_core=core[core.pool=='main']
    if len(main_core)!=3539:raise ValueError('Existing main benchmark changed')
    expected=set(main_core.cut_id)
    hashes={}
    for name in features:
        path=work_dir('features')/(name+'.npz')
        with np.load(path,allow_pickle=False) as z:
            if not expected.issubset(set(z['cut_id'].astype(str))):raise ValueError('Incomplete feature: '+name)
        hashes[name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (work_dir('actual_reference_run')/'evaluation_feature_hashes.json').write_text(json.dumps(hashes,indent=2))
    main_core.to_csv(work_dir('actual_reference_run')/'main_manifest.csv',index=False,encoding='utf-8-sig')
    main_core[main_core.split=='dev'].to_csv(work_dir('actual_reference_run')/'dev_manifest.csv',index=False,encoding='utf-8-sig')
    common=[str(PKG_ROOT/'src/b07_evaluate.py'),'--features',','.join(features),
            '--tasks','group3,type_single,type_multi,species','--no-random']
    run([*common,'--protocols','devcv,test','--boot','2000','--run','actual_ast_vggish_primary'],'evaluation_primary.log')
    run([*common,'--protocols','allcv,filecv','--boot','500','--run','actual_ast_vggish_secondary'],'evaluation_secondary.log')
    log(f'Complete matched actual-model evaluation; {time.time()-t0:.0f}s')
    return 0

if __name__=='__main__':raise SystemExit(main())
