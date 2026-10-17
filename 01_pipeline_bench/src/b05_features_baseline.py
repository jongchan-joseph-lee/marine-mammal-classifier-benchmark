# -*- coding: utf-8 -*-
"""5단계. 기준선 특징을 뽑는다. 사전학습 모델이 필요 없는 것만 다룬다.

만드는 특징 (work/features/<이름>.npz : cut_id, X, names, info)
  meta          log10 fs, log10 길이.  음원을 읽지 않는다. '녹음 조건만으로 얼마나 맞는가'를 재는 대조군이다.
  mfcc16k       16 kHz 로 맞춘 뒤 MFCC 20개의 평균·표준편차와 그 변화량(델타)의 평균·표준편차 (80차원).
                16 kHz 입력을 쓰는 방법(VGGish, AST 등)과 같은 대역 조건의 얕은 기준선이다.
                fs 가 16 kHz 보다 낮은 파일은 위 대역이 비게 된다. 이 빈 대역이 fs 를 알려 주는 지름길이 될 수 있다.
  mfcc_band4k   모든 파일을 8 kHz 로 맞춘 공통 대역(0-4 kHz) 조건의 MFCC (80차원). fs 가 8 kHz 미만인 파일은 뺀다.
                'fs 가 달라서 생기는 빈 대역'을 없앤 조건이다. 4 kHz 위의 성분(대부분의 휘슬과 클릭)은 버려진다.
  physv0        원래 fs 에서, 초와 Hz 단위로 정의한 형태 기술자 (51차원). 연구자 자신의 분류기를 위한 출발점 후보이며
                다듬지 않은 초안이다. 정의는 PHYS_NAMES 와 phys_features() 의 주석에 있다.

문헌의 방법이 아니라 이 묶음에서 고른 구현이다: 창 길이, 멜 대역 수, 요약 통계, physv0 의 모든 기준값.
한 컷에서 앞 120초까지만 쓴다(wb_common.MAX_ANALYSIS_S). 핵심 표본은 60초 이하라 해당이 없다.

쓰는 법  python src/b05_features_baseline.py [--set core|gold|all] [--only meta,mfcc16k,...]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from fractions import Fraction

import numpy as np
import pandas as pd

from b04_scan_audio_qc import NOISE_RISE_DB, RISE_MARGIN_DB, TF_DB, nfft_for, stft_power
from wb_common import DATA_DIR, MAX_ANALYSIS_S, csv_arg, index_audio, load_config, log, read_audio, work_dir

# ----------------------------------------------------------------------------------------------
# MFCC
# ----------------------------------------------------------------------------------------------
def resample_to(x: np.ndarray, fs: int, target: int) -> np.ndarray:
    from scipy.signal import resample_poly
    if fs == target:
        return x
    fr = Fraction(target, fs).limit_denominator(2000)
    return resample_poly(x, fr.numerator, fr.denominator)


def mel_filterbank(n_mels: int, nfft: int, fs: int, fmin: float, fmax: float) -> np.ndarray:
    hz2mel = lambda f: 2595.0 * np.log10(1.0 + f / 700.0)
    mel2hz = lambda m: 700.0 * (10 ** (m / 2595.0) - 1.0)
    pts = mel2hz(np.linspace(hz2mel(fmin), hz2mel(fmax), n_mels + 2))
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, ce, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (freqs - lo) / max(ce - lo, 1e-9)
        dn = (hi - freqs) / max(hi - ce, 1e-9)
        fb[i] = np.clip(np.minimum(up, dn), 0, None)
    return fb


def mfcc_summary(x: np.ndarray, fs: int, n_mels: int, fmin: float, fmax: float, n_mfcc: int = 20) -> np.ndarray:
    """25 ms 창, 10 ms 간격. c0(전체 레벨)는 녹음 이득에 좌우되므로 빼고 c1..c20 을 쓴다."""
    from scipy.fft import dct
    win, hop = int(round(0.025 * fs)), int(round(0.010 * fs))
    nfft = int(2 ** np.ceil(np.log2(win)))
    if len(x) < win:
        x = np.pad(x, (0, win - len(x)))
    n_frames = 1 + (len(x) - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    spec = np.abs(np.fft.rfft(x[idx] * np.hanning(win)[None, :], n=nfft, axis=1)) ** 2
    mel = spec @ mel_filterbank(n_mels, nfft, fs, fmin, fmax).T
    logmel = 10 * np.log10(mel + 1e-10)
    c = dct(logmel, type=2, axis=1, norm="ortho")[:, 1:n_mfcc + 1]
    if len(c) >= 3:
        d = np.gradient(c, axis=0)
    else:
        d = np.zeros_like(c)
    return np.concatenate([c.mean(0), c.std(0), d.mean(0), d.std(0)]).astype(np.float32)


MFCC_NAMES = [f"{stat}_{kind}{i}" for kind, stat in [("c", "mean"), ("c", "std"), ("d", "mean"), ("d", "std")]
              for i in range(1, 21)]

# ----------------------------------------------------------------------------------------------
# physv0: 원래 fs, 물리 단위
# ----------------------------------------------------------------------------------------------
THIRD_OCT = [10 ** (k / 10) for k in range(18, 49)]          # 63 Hz ... 63 kHz, 31개 대역
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


# ----------------------------------------------------------------------------------------------
def save(name: str, cut_ids: list[str], X: np.ndarray, names: list[str], info: dict) -> None:
    fdir = work_dir("features")
    np.savez_compressed(fdir / f"{name}.npz", cut_id=np.array(cut_ids), X=X.astype(np.float32),
                        names=np.array(names), info=json.dumps(info, ensure_ascii=False))
    log(f"  저장 {name}: {X.shape}, NaN 이 있는 행 {int(np.isnan(X).any(1).sum())}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core", choices=["core", "gold", "all"])
    ap.add_argument("--only", nargs="+", default="meta,mfcc16k,mfcc_band4k,physv0", help="쉼표로 잇거나 띄어 쓴다")
    args = ap.parse_args()
    want = set(csv_arg(args.only))

    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    sel = m[m.role == "core"] if args.set == "core" else (m[m.gold != ""] if args.set == "gold" else m)

    if "meta" in want:
        X = np.c_[np.log10(sel.fs_hz.astype(float)), np.log10(sel.dur_s.astype(float))]
        save("meta", sel.cut_id.tolist(), X, ["log10_fs", "log10_dur"], dict(desc="fs 와 길이만. 음원을 읽지 않는다"))
    audio_feats = want & {"mfcc16k", "mfcc_band4k", "physv0"}
    if not audio_feats:
        return 0

    idx = index_audio(load_config()["audio_root"])
    sel = sel[sel.cut_id.isin(idx)]
    log(f"음원을 찾은 대상 {len(sel)}개")
    ids, A, B, C, okB = [], [], [], [], []
    t0 = time.time()
    for i, r in enumerate(sel.itertuples(index=False)):
        try:
            x, fs = read_audio(idx[r.cut_id], MAX_ANALYSIS_S)    # 핵심 표본은 60초 이하라 잘리지 않는다
        except Exception as e:  # noqa: BLE001
            log(f"  읽기 실패 {r.cut_id}: {e}")
            continue
        x = x - x.mean()
        ids.append(r.cut_id)
        if "mfcc16k" in want:
            A.append(mfcc_summary(resample_to(x, fs, 16000), 16000, n_mels=64, fmin=50, fmax=8000))
        if "mfcc_band4k" in want:
            if fs >= 8000:
                B.append(mfcc_summary(resample_to(x, fs, 8000), 8000, n_mels=40, fmin=50, fmax=4000))
                okB.append(True)
            else:
                B.append(np.full(80, np.nan, dtype=np.float32))
                okB.append(False)
        if "physv0" in want:
            C.append(phys_features(x, fs))
        if (i + 1) % 500 == 0:
            log(f"  {i + 1}/{len(sel)}  {time.time() - t0:.0f}초")
    if "mfcc16k" in want:
        save("mfcc16k", ids, np.vstack(A), MFCC_NAMES,
             dict(desc="16 kHz, 멜 64개(50-8000 Hz), 25 ms/10 ms, c1-c20 의 평균·표준편차와 델타의 평균·표준편차"))
    if "mfcc_band4k" in want:
        keep = np.array(okB)
        save("mfcc_band4k", [c for c, k in zip(ids, keep) if k], np.vstack(B)[keep], MFCC_NAMES,
             dict(desc="공통 대역 조건: 8 kHz 로 맞춤(0-4 kHz), 멜 40개(50-4000 Hz). fs<8 kHz 파일 제외",
                  excluded_fs_lt_8k=int((~keep).sum())))
    if "physv0" in want:
        Xc = np.vstack(C)
        save("physv0", ids, Xc, PHYS_NAMES, dict(desc="원래 fs, 물리 단위 형태 기술자 초안 (51차원)"))
        pd.DataFrame(Xc, columns=PHYS_NAMES).assign(cut_id=ids).set_index("cut_id").to_csv(
            work_dir("features") / "physv0_table.csv", encoding="utf-8-sig")
    log(f"끝. {time.time() - t0:.0f}초")
    return 0


if __name__ == "__main__":
    sys.exit(main())
