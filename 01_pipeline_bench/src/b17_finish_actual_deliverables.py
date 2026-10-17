"""Wait for complete evaluations, export reports, and make a validated full update archive."""
from __future__ import annotations
import json, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
from wb_common import PKG_ROOT, work_dir, log

def main():
    started=time.time()
    marker=work_dir('eval','actual_ast_vggish_secondary')/'run_info.json'
    while not marker.exists():
        if time.time()-started>6000:raise TimeoutError('Complete evaluations not available')
        time.sleep(3)
    info=json.loads(marker.read_text())
    if info['n_core_used']!=3539 or info['n_core_dropped']!=0:raise ValueError('Partial evaluation cannot be delivered')
    for name in ['ast_zero','vggish','vggish_head10p24']:
        with np.load(work_dir('features')/(name+'.npz'),allow_pickle=False) as z:
            if z['X'].shape[0]!=3539 or not np.isfinite(z['X']).all():raise ValueError('Embedding incomplete: '+name)
    with (work_dir('actual_reference_run')/'report_export.log').open('w',encoding='utf-8') as output:
        subprocess.run([sys.executable,str(PKG_ROOT/'src/b16_export_actual_report.py')],stdout=output,
                       stderr=subprocess.STDOUT,check=True)
    output_dir=PKG_ROOT.parent/'deliverables';output_dir.mkdir(exist_ok=True)
    for filename in ['Watkins_AST_VGGish_Actual_Report.html','comparison_summary.csv','comparison.png']:
        shutil.copyfile(work_dir('actual_report')/filename,output_dir/filename)
    archive=output_dir/'Watkins_Bench_v1(1).zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in sorted(PKG_ROOT.rglob('*')):
            if not path.is_file():continue
            rel=path.relative_to(PKG_ROOT)
            if '__pycache__' in rel.parts or 'embedding_cache' in rel.parts:continue
            if path.name=='watkins_bench_config.local.json' or path.suffix=='.pyc':continue
            z.write(path,rel.as_posix())
    with zipfile.ZipFile(archive) as z:
        bad=z.testzip()
        if bad:raise ValueError('ZIP CRC failed: '+bad)
        names=set(z.namelist())
        required=['src/b06_embed_pretrained.py','src/vggish_reference.py','work/features/ast_zero.npz',
                  'work/features/vggish.npz','work/eval/actual_ast_vggish_primary/results.csv',
                  'notebooks/Watkins_AST_VGGish_Actual_Results.ipynb']
        if not set(required).issubset(names):raise ValueError('Missing deliverable members')
    log('Actual results and full update archive ready: '+str(archive))
    return 0

if __name__=='__main__':raise SystemExit(main())
