"""Run full-cohort analyses after each requested embedding array is saved.

Waiting is bounded, progress is printed, and every failed subprocess stops the
pipeline. This can run alongside embedding inference, with one CPU thread for
analysis to limit competition. It never creates partial full-cohort results.
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default='ast_zero,ast_tile')
    ap.add_argument('--wait-timeout',type=float,default=10800)
    args=ap.parse_args()
    env=os.environ.copy()
    for key in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS']:
        env[key]='1'
    records=[];target=ROOT/'output/provenance/resumed_analysis_stages.json'
    for feature in args.features.split(','):
        start=time.monotonic();last_message=-60.
        while not (ROOT/'output/features'/(feature+'.npz')).exists():
            elapsed=time.monotonic()-start
            if elapsed>args.wait_timeout:raise TimeoutError(f'{feature}: no complete array after {elapsed:.0f}s; saved calculations are preserved')
            if elapsed-last_message>=60:
                print(f'{feature}: waiting for the complete saved array ({elapsed:.0f}s)',flush=True);last_message=elapsed
            time.sleep(5)
        stages=[('stage1_structure.py',[]),('stage2_sampling.py',[]),('stage3_five_analyses.py',['--stage','all'])]
        for script,extra in stages:
            stamp=time.time();command=[sys.executable,'-u',str(ROOT/'scripts'/script),'--features',feature,*extra]
            print(f'{feature}: starting {script}',flush=True)
            subprocess.run(command,cwd=ROOT.parent,env=env,check=True)
            records.append(dict(feature=feature,script=script,started_utc_epoch=stamp,seconds=time.time()-stamp,completed=True))
            temporary=target.with_suffix('.json.pending');temporary.write_text(json.dumps(records,indent=2));temporary.replace(target)
    print('All requested completed-feature analyses finished.',flush=True)

if __name__=='__main__':main()
