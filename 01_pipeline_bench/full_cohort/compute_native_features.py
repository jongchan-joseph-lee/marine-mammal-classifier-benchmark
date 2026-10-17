"""Reuse frozen core features and compute only missing full-cohort rows.

The original physv0 and MFCC routines are imported unchanged. Each new result
has a per-cut checkpoint; final arrays require all 10,694 eligible main IDs.
"""
from pathlib import Path
import concurrent.futures as cf
import hashlib
import json
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT.parent
SOURCE = BASE / 'deep_review/raw_new/src'
sys.path.insert(0, str(SOURCE))
from b05_features_baseline import phys_features, resample_to, mfcc_summary, PHYS_NAMES, MFCC_NAMES
from wb_common import read_audio


def main():
    m = pd.read_csv(BASE / 'deep_review/raw_new/data/bench_manifest.csv', dtype={'cut_id':str}, low_memory=False).fillna('')
    m = m[m.pool.eq('main') & m.hold_reason.eq('')].sort_values('cut_id').reset_index(drop=True)
    assert len(m) == 10694
    out = ROOT / 'output/features'
    out.mkdir(parents=True, exist_ok=True)
    prov = ROOT / 'output/provenance'
    prov.mkdir(parents=True, exist_ok=True)
    cache = ROOT / 'work/native_cache'
    cache.mkdir(parents=True, exist_ok=True)
    token = hashlib.sha256((SOURCE / 'b05_features_baseline.py').read_bytes() + (SOURCE / 'b04_scan_audio_qc.py').read_bytes()).hexdigest()
    names = {'physv0':PHYS_NAMES, 'mfcc16k':MFCC_NAMES}
    frozen = {}
    frozen_paths = {}
    for f in names:
        p = BASE / f'deep_review/raw_new/work/features/{f}.npz'
        z = np.load(p, allow_pickle=False)
        assert list(z['names']) == names[f]
        frozen[f] = {c:x for c,x in zip(z['cut_id'].astype(str),z['X'])}
        frozen_paths[f] = {'file':str(p.relative_to(BASE)), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    ids = m.cut_id.to_numpy(dtype=str)
    values = {f:{c:frozen[f][c] for c in ids if c in frozen[f]} for f in names}
    assert len(values['physv0']) == len(values['mfcc16k']) == 3539
    meta = np.c_[np.log10(m.fs_hz.astype(float)), np.log10(m.dur_s.astype(float))].astype(np.float32)
    np.savez_compressed(out / 'meta.npz', cut_id=ids, X=meta, names=np.array(['log10_fs','log10_dur']),
                        info=json.dumps({'cohort_n':len(m),'audio_free_control':True}))
    m.to_csv(prov / 'main_full_metadata.csv', index=False)
    start = time.time()

    def compute(row):
        path = ROOT / 'audio_original' / (row.cut_id + '.flac')
        x, fs = read_audio(str(path), 120.)
        assert fs == int(row.fs_hz)
        x = x - x.mean()
        return {'physv0':phys_features(x, fs),
                'mfcc16k':mfcc_summary(resample_to(x, fs, 16000),16000,n_mels=64,fmin=50,fmax=8000)}

    # Verify cache compatibility against actual waveforms before importing all
    # existing rows. The first ready core rows are used for a numerical check,
    # never to select a feature or compare labels/performance.
    checks = []
    while len(checks) < 12:
        ready = [r for r in m.itertuples(index=False)
                 if r.cut_id in values['physv0'] and (ROOT / 'audio_original' / (r.cut_id+'.flac')).exists()]
        if len(ready) < 12:
            time.sleep(2)
            continue
        for r in ready[:12]:
            actual = compute(r)
            for f in names:
                a = frozen[f][r.cut_id]
                b = actual[f]
                same_nan = np.array_equal(np.isnan(a), np.isnan(b))
                difference = float(np.nanmax(abs(a-b))) if np.isfinite(a).any() else 0.
                ok = same_nan and np.allclose(a,b,rtol=2e-5,atol=2e-5,equal_nan=True)
                checks.append({'cut_id':r.cut_id,'feature':f,'fs_hz':int(r.fs_hz),'max_abs':difference,'nan_mask_equal':same_nan,'passed':bool(ok)})
                if not ok: raise RuntimeError(f'Existing {f} cache is not numerically compatible: {r.cut_id}, {difference}')
        break
    (prov / 'native_cache_equivalence.json').write_text(json.dumps(checks,indent=2))
    print(f'Native cache verified on {len(checks)//2} real cuts. Reusing 3539 rows per feature.',flush=True)
    remaining = {r.cut_id:r for r in m.itertuples(index=False) if r.cut_id not in values['physv0']}
    resumed = computed = 0
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        while remaining:
            ready = [cid for cid in remaining if (ROOT / 'audio_original' / (cid+'.flac')).exists()][:80]
            if not ready:
                status = prov / 'audio_validation.json'
                if status.exists() and json.loads(status.read_text())['failed']:
                    raise RuntimeError(f'{len(remaining)} cuts remain and original audio download has failures; refusing a partial result')
                time.sleep(2)
                continue
            jobs = {}
            for cid in ready:
                row = remaining[cid]
                checkpoint = cache / (cid+'.npz')
                if checkpoint.exists():
                    try:
                        z = np.load(checkpoint,allow_pickle=False)
                        if str(z['token'].item()) == token+row.md5_b64:
                            for f in names: values[f][cid] = z[f]
                            resumed += 1
                            del remaining[cid]
                            continue
                    except (OSError,ValueError,KeyError): pass
                jobs[pool.submit(compute,row)] = (cid, row.md5_b64, checkpoint)
            for future in cf.as_completed(jobs):
                cid, md5, checkpoint = jobs[future]
                result = future.result()
                for f in names: values[f][cid] = result[f]
                temp = checkpoint.with_suffix('.npz.pending')
                with temp.open('wb') as stream:
                    np.savez_compressed(stream,token=token+md5,**result)
                temp.replace(checkpoint)
                computed += 1
                del remaining[cid]
            elapsed = time.time()-start
            progress = dict(expected=len(m),reused_core=3539,computed=computed,resumed_new=resumed,
                            ready_rows=len(values['physv0']),remaining=len(remaining),seconds=elapsed)
            (prov / 'native_progress.json').write_text(json.dumps(progress,indent=2))
            print(f'Native features {len(values["physv0"])}/{len(m)}; new {computed}; checkpoint reuse {resumed}; {elapsed:.1f}s',flush=True)
    for f in names:
        X = np.stack([values[f][c] for c in ids]).astype(np.float32)
        assert X.shape == (len(m),len(names[f])) and not np.isinf(X).any()
        info = dict(cohort_n=len(m),reused_core=3539,computed_new=computed,resumed_new=resumed,
                    unchanged_original_routine=True,max_analysis_seconds=120,n_long_cuts_truncated=int((m.dur_s>120).sum()),
                    rows_all_nan=int(np.isnan(X).all(1).sum()),rows_some_nan=int(np.isnan(X).any(1).sum()),
                    native_rate_preserved=(f=='physv0'),source_code_sha256=token,frozen_source=frozen_paths[f])
        temp = out / (f+'.npz.pending')
        with temp.open('wb') as stream:
            np.savez_compressed(stream,cut_id=ids,X=X,names=np.array(names[f]),info=json.dumps(info))
        temp.replace(out / (f+'.npz'))
    progress.update(complete=True)
    (prov / 'native_progress.json').write_text(json.dumps(progress,indent=2))
    print('Complete full-cohort native feature arrays saved; no eligible cut omitted.',flush=True)


if __name__ == '__main__':
    main()
