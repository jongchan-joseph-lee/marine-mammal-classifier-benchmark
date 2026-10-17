"""Fetch verified original-rate FLACs for the complete eligible main cohort.

Resumable; preserves exclusion flags and the old source split. No waveform
resampling is performed. Validation checks published MD5 and native headers.
"""
from pathlib import Path
import argparse
import base64
import concurrent.futures as cf
import hashlib
import json
import time
import threading

import pandas as pd
import requests
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=12)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()
    m = pd.read_csv(BASE / 'deep_review/raw_new/data/bench_manifest.csv', dtype={'cut_id':str}, low_memory=False).fillna('')
    m = m[m.pool.eq('main') & m.hold_reason.eq('')].sort_values('cut_id')
    assert len(m) == 10694
    if args.limit:
        m = m.head(args.limit)
    public = json.loads((ROOT / 'input/public_manifest.json').read_text())
    index = {Path(x['path']).stem:x for x in public['audio_hashes'] if x['path'].endswith('.flac')}
    dest = ROOT / 'audio_original'
    out = ROOT / 'output/provenance'
    dest.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    start = time.time()
    local = threading.local()

    def one(row):
        entry = index[row.cut_id]
        expected = entry['hash'].split(':', 1)[1]
        assert expected == row.md5_b64, row.cut_id
        path = dest / (row.cut_id + '.flac')
        def verified():
            if not path.exists(): return False
            md5 = base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode()
            if md5 != expected: return False
            h = sf.info(path)
            if h.samplerate != int(row.fs_hz) or h.frames != int(row.n_frames):
                raise ValueError(f'Native sample-rate/frame mismatch: {row.cut_id}')
            return True
        if verified():
            return dict(cut_id=row.cut_id, status='reused', bytes=path.stat().st_size, md5_ok=True, native_header_ok=True)
        url = 'https://storage.googleapis.com/' + entry['path'][5:]
        error = None
        for attempt in range(3):
            try:
                if not hasattr(local, 'session'):
                    local.session = requests.Session()
                response = local.session.get(url, timeout=(15,45))
                response.raise_for_status()
                payload = response.content
                md5 = base64.b64encode(hashlib.md5(payload).digest()).decode()
                if md5 != expected: raise ValueError('MD5 mismatch')
                temp = path.with_suffix('.flac.pending')
                temp.write_bytes(payload)
                temp.replace(path)
                assert verified()
                return dict(cut_id=row.cut_id, status='downloaded', bytes=len(payload), md5_ok=True, native_header_ok=True)
            except Exception as exc:
                error = f'{type(exc).__name__}: {exc}'
                if attempt < 2: time.sleep(attempt + 1)
        return dict(cut_id=row.cut_id, status='failed', error=error, md5_ok=False, native_header_ok=False, bytes=0)

    rows = []
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(one, r) for r in m.itertuples(index=False)]
        for f in cf.as_completed(futures):
            rows.append(f.result())
            if len(rows) % 100 == 0 or len(rows) == len(m):
                table = pd.DataFrame(rows).sort_values('cut_id')
                table.to_csv(out / 'audio_validation_in_progress.csv', index=False)
                done = sum(r['md5_ok'] for r in rows)
                seconds = time.time() - start
                print(f'Native originals {len(rows)}/{len(m)}; verified {done}; failed {len(rows)-done}; {seconds:.1f}s', flush=True)
    table = pd.DataFrame(rows).sort_values('cut_id')
    table.to_csv(out / 'audio_validation.csv', index=False)
    summary = dict(requested=len(m), verified=int(table.md5_ok.sum()), failed=int((~table.md5_ok).sum()),
                   bytes=int(table.bytes.sum()), seconds=time.time()-start, dataset_hash=public['dataset_hash'],
                   native_rates_preserved=True, max_audio_analysis_seconds=120)
    (out / 'audio_validation.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)
    return bool(summary['failed'])


if __name__ == '__main__':
    raise SystemExit(main())
