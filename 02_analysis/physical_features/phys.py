from __future__ import annotations
import numpy as np
import math
from fractions import Fraction
try:
    from scipy import signal
    import scipy
except Exception: pass
WIN_S = 0.032

RISE_DB = 6.0

NOISE_RISE_DB = 10 * np.log10(np.log(20) / np.log(2))

RISE_MARGIN_DB = 3.0

TF_DB = 9.0

LINE_DB = 8.0

LINE_TIME_FRAC = 0.90

def nfft_for(fs: int, n: int) -> int:
    """창 길이를 초로 정하고 샘플 수는 fs 에 맞춘다. 짧은 파일은 창을 줄인다."""
    k = int(2 ** round(np.log2(max(fs * WIN_S, 64))))
    while k > 64 and n < 4 * k:
        k //= 2
    return int(min(max(k, 64), 8192))

def stft_power(x: np.ndarray, nfft: int, block: int = 1024) -> np.ndarray:
    """(프레임 × 주파수 칸) 파워. Hann 창, 50 % 겹침. 프레임을 block 개씩 나눠 계산해 메모리를 아낀다(값은 같다)."""
    hop = nfft // 2
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    n_frames = 1 + (len(x) - nfft) // hop
    win = np.hanning(nfft)[None, :]
    base = np.arange(nfft)[None, :]
    out = np.empty((n_frames, nfft // 2 + 1))
    for s in range(0, n_frames, block):
        e = min(s + block, n_frames)
        idx = base + hop * np.arange(s, e)[:, None]
        out[s:e] = np.abs(np.fft.rfft(x[idx] * win, axis=1)) ** 2
    return out

MFCC_NAMES = [f"{stat}_{kind}{i}" for kind, stat in [("c", "mean"), ("c", "std"), ("d", "mean"), ("d", "std")]
              for i in range(1, 21)]

THIRD_OCT = [10 ** (k / 10) for k in range(18, 49)]

PHYS_NAMES = (
    ["log_sig_f05", "log_sig_f50", "log_sig_f95", "sig_bw_oct", "sig_bins_frac", "rise_peak_db", "tf_active_frac"]
    + [f"rise3oct_{int(round(f))}" for f in THIRD_OCT]
    + ["duty", "log1p_events_per_s", "log_event_dur_med", "log_event_dur_max"]
    + ["tonal_conc", "log1p_fm_oct_per_s", "pk_range_oct"]
    + ["log_tr_kurt", "log1p_tr_clicks_per_s", "tr_log_ici_med", "tr_ici_cv"]
    + ["mod_lag_s", "mod_strength"]
)

def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """True 가 이어진 구간의 (시작, 끝+1) 목록."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))

def phys_features(x: np.ndarray, fs: int) -> np.ndarray:
    """형태 기술자 51개. 계산할 수 없는 값은 NaN 으로 둔다 (평가 단계에서 겹 안의 중앙값으로 채운다).

    A. 사건 대역 (7)   칸마다 P95/P50 이 잡음 기대값(6.4 dB)+3 dB 를 넘는 만큼을 무게로 한 누적 주파수 5·50·95 % [log10 Hz],
                       그 폭 [옥타브], 솟은 칸의 비율, 가장 크게 솟은 정도 [dB], 솟은 시간-주파수 칸의 비율
    B. 1/3 옥타브 솟음 (31)  63 Hz-63 kHz 대역마다 솟은 정도의 평균 [dB]. 나이퀴스트 위나 분해능 아래 대역은 0
                       (녹음되지 않은 대역을 '소리가 없었다'와 같은 0 으로 두는 것은 한계다. 공통 대역 조건과 함께 본다)
    C. 시간 구조 (4)   활동 프레임의 비율, 초당 사건 수 [log(1+x)], 사건 길이의 중앙값과 최댓값 [log10 초]
    D. 음조성 (3)      활동 프레임에서 솟은 파워가 가장 센 칸 주변 3칸에 모인 비율(휘슬은 크고 클릭은 작다),
                       최대 칸 주파수의 프레임 간 변화 [옥타브/초, log(1+x)], 그 범위(사분위 폭) [옥타브]
    E. 충격성 (4)      1 ms 창으로 다시 본 포락선의 첨도 [log10], 초당 클릭 수 [log(1+x)],
                       클릭 간격의 중앙값 [log10 초]과 변동계수
    F. 반복성 (2)      활동 포락선 자기상관의 첫 봉우리 지연 [초]과 높이
    """
    out = np.full(len(PHYS_NAMES), np.nan, dtype=np.float64)
    n = len(x)
    if n < 128:
        return out.astype(np.float32)
    x = x - x.mean()
    nfft = nfft_for(fs, n)
    hop_s = (nfft // 2) / fs
    P = stft_power(x, nfft)
    T = P.shape[0]
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    lo = freqs >= max(10.0, 2 * fs / nfft)
    if T < 8:
        return out.astype(np.float32)
    p50 = np.percentile(P, 50, axis=0) + 1e-30
    p95 = np.percentile(P, 95, axis=0)
    rise = 10 * np.log10((p95 + 1e-30) / p50)
    rs = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(rise, 1, mode="edge"), 3), axis=1)
    wgt = np.clip(rs - (NOISE_RISE_DB + RISE_MARGIN_DB), 0, None)
    wgt[~lo] = 0

    # A
    if wgt.sum() > 0:
        c = np.cumsum(wgt) / wgt.sum()
        f05, f50, f95 = (float(freqs[min(np.searchsorted(c, q), len(freqs) - 1)]) for q in (0.05, 0.5, 0.95))
        out[0:4] = [np.log10(max(f05, 1)), np.log10(max(f50, 1)), np.log10(max(f95, 1)),
                    np.log2(max(f95, 1) / max(f05, 1))]
    out[4] = float((wgt[lo] > 0).mean())
    out[5] = float(rs[lo].max())
    W = P / p50[None, :]
    hot = (W > 10 ** (TF_DB / 10)) & lo[None, :]
    out[6] = float(hot[:, lo].mean())

    # B
    for j, fc in enumerate(THIRD_OCT):
        band = (freqs >= fc * 2 ** (-1 / 6)) & (freqs < fc * 2 ** (1 / 6)) & lo
        out[7 + j] = float(wgt[band].mean()) if band.any() else 0.0

    # C
    a = hot[:, lo].mean(1)                         # 프레임마다 솟은 칸의 비율
    act = a > 0.02                                 # 잡음 기대값(약 0.004)의 다섯 배
    runs = _runs(act)
    merged = []
    for s, e in runs:                              # 한 프레임 이하의 틈은 잇는다
        if merged and s - merged[-1][1] <= 1:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    dur_total = max(T * hop_s, 1e-9)
    out[38] = float(act.mean())
    out[39] = float(np.log1p(len(merged) / dur_total))
    if merged:
        durs = np.array([(e - s) * hop_s for s, e in merged])
        out[40] = float(np.log10(np.median(durs)))
        out[41] = float(np.log10(durs.max()))

    # D
    if act.sum() >= 2:
        E = np.clip(W - 1, 0, None) * lo[None, :]
        Ea = E[act]
        pk = Ea.argmax(1)
        tot = Ea.sum(1) + 1e-30
        idx3 = np.clip(pk[:, None] + np.array([-1, 0, 1])[None, :], 0, Ea.shape[1] - 1)
        out[42] = float(np.median(np.take_along_axis(Ea, idx3, axis=1).sum(1) / tot))
        fpk = np.log2(np.maximum(freqs[pk], freqs[1]))
        ai = np.flatnonzero(act)
        adj = np.diff(ai) == 1                     # 이웃한 활동 프레임끼리만
        if adj.any():
            out[43] = float(np.log1p(np.median(np.abs(np.diff(fpk))[adj]) / hop_s))
        out[44] = float(np.percentile(fpk, 75) - np.percentile(fpk, 25))

    # E: 1 ms 창 (fs 가 낮으면 32 샘플 창)
    n2 = int(2 ** round(np.log2(max(fs * 0.001, 32))))
    if n >= 8 * n2 and fs >= 8000:
        P2 = stft_power(x, n2)
        f2 = np.fft.rfftfreq(n2, 1 / fs)
        lo2 = f2 >= max(500.0, 2 * fs / n2)        # 클릭은 고주파 성분으로 본다
        if lo2.sum() >= 2 and P2.shape[0] >= 16:
            W2 = P2[:, lo2] / (np.median(P2[:, lo2], axis=0)[None, :] + 1e-30)
            env = W2.mean(1)
            m, sd = env.mean(), env.std() + 1e-30
            out[45] = float(np.log10(((env - m) ** 4).mean() / sd ** 4))
            thr = 8.0 * np.median(env)
            pk_mask = (env > thr) & (env >= np.roll(env, 1)) & (env > np.roll(env, -1))
            t_pk = np.flatnonzero(pk_mask) * (n2 // 2) / fs
            out[46] = float(np.log1p(len(t_pk) / (len(env) * (n2 // 2) / fs)))
            if len(t_pk) >= 3:
                ici = np.diff(t_pk)
                out[47] = float(np.log10(np.median(ici)))
                out[48] = float(ici.std() / (ici.mean() + 1e-30))

    # F
    if T >= 16 and a.std() > 0:
        z = a - a.mean()
        ac = np.correlate(z, z, mode="full")[T - 1:]
        ac = ac / (ac[0] + 1e-30)
        first_neg = np.flatnonzero(ac < 0)
        if len(first_neg) and first_neg[0] < len(ac) - 2:
            seg = ac[first_neg[0]:]
            k = int(seg.argmax())
            out[49] = float((first_neg[0] + k) * hop_s)
            out[50] = float(seg[k])
    return out.astype(np.float32)