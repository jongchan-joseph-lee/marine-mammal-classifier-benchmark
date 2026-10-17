"""Fetch original FLACs for the existing benchmark; verify hash and native headers."""
from __future__ import annotations
import argparse, base64, concurrent.futures, hashlib, json, time
from pathlib import Path
import pandas as pd
import requests
import soundfile as sf
from wb_common import DATA_DIR, load_config, work_dir, log

BASE = 'https://storage.googleapis.com/esp-data-274503/watkins/v0.1.0/'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=20)
    ap.add_argument('--split', choices=['all','dev','test'], default='all')
    args = ap.parse_args()
    dest = Path(load_config()['audio_root']); dest.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(DATA_DIR/'bench_manifest.csv',dtype=str,keep_default_na=False)
    selected = manifest[(manifest.role=='core') & (manifest.pool=='main')]
    if args.split != 'all': selected = selected[selected.split==args.split]
    response = requests.get(BASE+'manifest.json',timeout=60); response.raise_for_status()
    public = response.json()
    index = {Path(x['path']).stem:x for x in public['audio_hashes'] if x['path'].endswith('.flac')}
    started=time.time()
    def one(row):
        cid = row.cut_id; remote=index[cid]; expected=remote['hash'].split(':',1)[1]
        if expected != row.md5_b64: raise ValueError('Public and benchmark MD5 disagree: '+cid)
        path=dest/(cid+'.flac')
        def valid():
            if not path.exists(): return False
            digest=base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
            if digest != expected: return False
            header=sf.info(path)
            if header.samplerate != int(row.fs_hz) or header.frames != int(row.n_frames):
                raise ValueError('Native header mismatch: '+cid)
            return True
        if valid(): return dict(cut_id=cid,status='reused',md5_ok=True,header_ok=True,bytes=path.stat().st_size)
        url='https://storage.googleapis.com/'+remote['path'][5:]
        error=None
        for attempt in range(3):
            try:
                r=requests.get(url,timeout=(15,90));r.raise_for_status()
                digest=base64.b64encode(hashlib.md5(r.content).digest()).decode()
                if digest != expected: raise ValueError('Downloaded MD5 mismatch: '+cid)
                temp=path.with_suffix('.flac.tmp');temp.write_bytes(r.content);temp.replace(path)
                if not valid(): raise ValueError('Post-transfer verification failed: '+cid)
                return dict(cut_id=cid,status='downloaded',md5_ok=True,header_ok=True,bytes=len(r.content))
            except Exception as exc:
                error=str(exc)
                if attempt<2: time.sleep(attempt+1)
        return dict(cut_id=cid,status='failed',md5_ok=False,header_ok=False,error=error)
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,r) for r in selected.itertuples(index=False)]
        for f in concurrent.futures.as_completed(futures):
            rows.append(f.result())
            if len(rows)%100==0 or len(rows)==len(futures):
                log(f'Original audio {len(rows)}/{len(futures)}; {time.time()-started:.0f}s; failed {sum(x["status"]=="failed" for x in rows)}')
    table=pd.DataFrame(rows).sort_values('cut_id')
    out=work_dir('actual_reference_run');table.to_csv(out/('audio_validation_'+args.split+'.csv'),index=False)
    summary=dict(n_requested=len(selected),n_verified=int(table.md5_ok.sum()),n_failed=int((table.status=='failed').sum()),
                 original_sample_rates_preserved=True,source_dataset_hash=public['dataset_hash'],seconds=round(time.time()-started,2))
    (out/('audio_validation_'+args.split+'.json')).write_text(json.dumps(summary,indent=2),encoding='utf-8')
    log(json.dumps(summary));return bool(summary['n_failed'])

if __name__=='__main__':raise SystemExit(main())
