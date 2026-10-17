"""Tonal vs pulsed candidate features at the original sample rate.
MOD  : envelope modulation spectrum (100 Hz .. min(5 kHz, fs/4)) - peak prominence [dB], peak freq (PRR est.), modulation depth
CPP  : cepstral peak prominence in quefrency 0.2-10 ms (PRR / f0 100-5000 Hz), dB above regression line, and its quefrency
EKURT: kurtosis of the full-resolution Hilbert envelope inside active frames (pulses -> peaky)
Frames 50 ms / hop 25 ms; features = median over the 30 % most energetic frames."""
import numpy as np, soundfile as sf, pandas as pd, sys
from scipy.signal import butter, sosfiltfilt, hilbert
from multiprocessing import Pool
def feats(cid):
    try:
        x, fs = sf.read(f'/home/claude/audio_core/{cid}.flac')
        if x.ndim > 1: x = x[:, 0]
        x = x - x.mean()
        sos = butter(4, 100, 'hp', fs=fs, output='sos'); x = sosfiltfilt(sos, x)
        env_all = np.abs(hilbert(x))
        N = int(0.05 * fs); H = N // 2
        if len(x) < 2 * N: return cid, [np.nan] * 7
        starts = np.arange(0, len(x) - N, H)
        en = np.array([np.sum(x[s:s+N]**2) for s in starts])
        top = starts[en >= np.quantile(en, 0.7)]
        w = np.hanning(N)
        fmod = np.fft.rfftfreq(N, 1/fs); hi = min(5000, fs/4); mb = (fmod >= 100) & (fmod <= hi)
        q = np.arange(N) / fs; qmin = 1 / min(5000, fs/2.5); qb = (q >= qmin) & (q <= 0.010)
        P, PF, MD, C, CQ, K = [], [], [], [], [], []
        for s in top:
            seg = x[s:s+N]; e = env_all[s:s+N]
            em = e.mean() + 1e-12
            S = np.abs(np.fft.rfft((e - em) * w))**2
            Sm = S[mb]
            P.append(10*np.log10(Sm.max() / (np.median(Sm) + 1e-30)))
            PF.append(fmod[mb][Sm.argmax()])
            MD.append(np.sqrt(np.sum(S[mb]) / (np.sum(w**2))) / em)   # rms modulation in band / mean envelope
            L = 10*np.log10(np.abs(np.fft.fft(seg * w))**2 + 1e-20)
            cep = np.real(np.fft.ifft(L)); cq = cep[qb]; qq = q[qb]
            a, b = np.polyfit(qq, cq, 1); r = cq - (a*qq + b); k = r.argmax()
            C.append(r[k]); CQ.append(qq[k])
            ee = e - e.mean(); K.append(np.mean(ee**4) / (np.mean(ee**2)**2 + 1e-30))
        return cid, [np.median(P), np.median(PF), np.median(MD), np.median(C), np.median(CQ), np.log10(np.median(K)), fs]
    except Exception as ex:
        return cid, [np.nan] * 7
if __name__ == '__main__':
    d = pd.read_csv('/home/claude/pkg_in/data/bench_manifest.csv', low_memory=False)
    m = d[(d.role == 'core') & d.task_type_single.isin(['whistle', 'moan', 'pulsed_call', 'growl', 'click', 'pulse_train'])]
    ids = m.cut_id.astype(str).tolist()
    with Pool(8) as p: res = p.map(feats, ids, chunksize=8)
    out = pd.DataFrame([r for _, r in res], columns=['mod_peak_db', 'mod_peak_hz', 'mod_depth', 'cpp_db', 'cpp_q', 'env_kurt_log', 'fs'], index=[c for c, _ in res])
    out.index.name = 'cut_id'; out.to_csv('tp_feats.csv'); print(out.describe().T.round(3))
