import numpy as np, pandas as pd, json, soundfile as sf, io, base64
from scipy.signal import stft
from PIL import Image
z = np.load('/home/claude/pkg_in/work/features/physv0.npz', allow_pickle=True)
names = [str(n) for n in z['names']]; ids = z['cut_id'].astype(str); X = pd.DataFrame(z['X'].astype(float), index=ids, columns=names)
d = pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv', low_memory=False); d['cut_id'] = d.cut_id.astype(str); D = d.set_index('cut_id')
m = D.loc[ids]
TYPES = ['whistle', 'moan', 'click', 'pulse_train', 'pulsed_call', 'growl']
DECK = {'whistle': '5702601X', 'moan': '8800601U', 'click': '9420108A', 'pulse_train': '6002602W', 'pulsed_call': '9420108L', 'growl': '6700900B'}
KEY = [  # name, label, unit, transform(stored)->display, display format
    ('log_sig_f05', 'Signal band low (f05)', 'kHz', 'p10k'), ('log_sig_f50', 'Centre frequency (f50)', 'kHz', 'p10k'),
    ('log_sig_f95', 'Signal band high (f95)', 'kHz', 'p10k'), ('sig_bw_oct', 'Bandwidth (f05–f95)', 'oct', 'id'),
    ('tonal_conc', 'Tonal concentration (TC)', '', 'id'), ('log1p_fm_oct_per_s', 'FM rate of peak', 'oct/s', 'expm1'),
    ('pk_range_oct', 'Peak-frequency range (IQR)', 'oct', 'id'), ('duty', 'Active-frame fraction', '', 'id'),
    ('log_event_dur_med', 'Event duration (median)', 's', 'p10'), ('log1p_events_per_s', 'Events per second', '/s', 'expm1'),
    ('log_tr_kurt', 'Envelope kurtosis (1 ms)', '', 'p10'), ('log1p_tr_clicks_per_s', 'Clicks per second', '/s', 'expm1'),
    ('tr_log_ici_med', 'Inter-click interval (median)', 'ms', 'p10ms'), ('tr_ici_cv', 'ICI variation (CV)', '', 'id'),
    ('mod_lag_s', 'Repetition lag (autocorr.)', 's', 'id'), ('mod_strength', 'Repetition strength', '', 'id'), ('rise_peak_db', 'Max band SNR', 'dB', 'id')]
def disp(v, t):
    if v is None or np.isnan(v): return None
    return {'p10k': 10 ** v / 1000, 'id': v, 'expm1': np.expm1(v), 'p10': 10 ** v, 'p10ms': 10 ** v * 1000}[t]
single = m[m.task_type_single.isin(TYPES)]
K = [k[0] for k in KEY]
Z = X.loc[single.index, K]; Zs = (Z - Z.median()) / (Z.quantile(.75) - Z.quantile(.25) + 1e-9); Zs = Zs.fillna(0)
def tile(cid, W=360, H=150, cap=10.0):
    x, fs = sf.read(f'/home/claude/audio_core/{cid}.flac'); x = x[:, 0] if x.ndim > 1 else x; x = x[:int(cap * fs)]
    n = int(2 ** np.round(np.log2(max(fs * 0.012, 128)))); f, t, Zx = stft(x, fs, nperseg=n, noverlap=int(n * 0.75))
    P = np.abs(Zx) ** 2; L = 10 * np.log10(P + 1e-20); hi = np.percentile(L, 99.7); L = np.clip((L - (hi - 55)) / 55, 0, 1)
    img = Image.fromarray((L[::-1] * 255).astype(np.uint8)).resize((W, H), Image.BILINEAR)
    b = io.BytesIO(); img.save(b, 'PNG', optimize=True); return base64.b64encode(b.getvalue()).decode(), fs, len(x) / fs
ex = {}
for ty in TYPES:
    sub = single[single.task_type_single == ty]; med = Zs.loc[sub.index].median()
    dist = ((Zs.loc[sub.index] - med) ** 2).sum(1).sort_values()
    picks = [DECK[ty]]; sps = {D.loc[DECK[ty], 'species']}; srcs = {D.loc[DECK[ty], 'g_site']}
    for c in dist.index:
        if len(picks) == 3: break
        if D.loc[c, 'species'] in sps or D.loc[c, 'g_site'] in srcs: continue
        picks.append(c); sps.add(D.loc[c, 'species']); srcs.add(D.loc[c, 'g_site'])
    lst = []
    for c in picks:
        b64, fs, dur = tile(c)
        vals = {k[0]: (None if disp(X.loc[c, k[0]], k[3]) is None else round(float(disp(X.loc[c, k[0]], k[3])), 4)) for k in KEY}
        lst.append(dict(id=c, sp=D.loc[c, 'species_ko'], spe=D.loc[c, 'species_en'], tx=D.loc[c, 'group3'], fs=int(fs), dur=round(dur, 2),
                        site=D.loc[c, 'g_site'], note=str(D.loc[c, 'note'])[:160], img=b64, vals=vals, deck=(c == DECK[ty]), pct=round(float((dist.rank(pct=True)).get(c, np.nan)), 2) if c in dist.index else None))
    ex[ty] = {'n': int(len(sub)), 'n_species': int(sub.species.nunique()), 'n_sites': int(sub.g_site.nunique()), 'examples': lst,
              'tx': {g: int((sub.group3 == g).sum()) for g in ['Odontoceti', 'Mysticeti', 'Pinnipedia']}}
# distributions on stored scale with display ticks
dist_out = {}
for k, lab, unit, t in KEY:
    v = X.loc[single.index, k]; lo, hi = np.nanpercentile(v, 1), np.nanpercentile(v, 99)
    if hi <= lo: hi = lo + 1
    bins = np.linspace(lo, hi, 31)
    h = {ty: np.histogram(np.clip(v[single.task_type_single == ty].dropna(), lo, hi), bins)[0].tolist() for ty in TYPES}
    med = {ty: (None if v[single.task_type_single == ty].dropna().empty else round(float(disp(np.nanmedian(v[single.task_type_single == ty]), t)), 4)) for ty in TYPES}
    nn = {ty: int(v[single.task_type_single == ty].notna().sum()) for ty in TYPES}
    dist_out[k] = dict(label=lab, unit=unit, t=t, lo=float(lo), hi=float(hi), h=h, med=med, n=nn)
json.dump({'types': TYPES, 'key': [dict(k=a, label=b, unit=c, t=e) for a, b, c, e in KEY], 'ex': ex, 'dist': dist_out, 'names': names}, open('phys_ex.json', 'w'), ensure_ascii=False)
import os; print(os.path.getsize('phys_ex.json') / 1e6, 'MB'); print({t: [e['id'] for e in ex[t]['examples']] for t in TYPES})
