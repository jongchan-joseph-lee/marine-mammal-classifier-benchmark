# -*- coding: utf-8 -*-
"""4단계. 음원을 원래 fs 그대로 읽어 품질과 '신호가 있는 대역'을 잰다. 읽기만 한다.

'참고래는 신호 품질이 나쁘다', '저주파(LF)가 지배적이다' 같은 판단을 귀와 눈 대신 같은 잣대로 재 보기 위한 단계다.
여기서 나온 값은 사람의 청취 판정을 대신하지 않는다. 검토할 파일을 고르는 데 쓴다.

재는 것 (파일마다 한 행)
  헤더      fs, 프레임 수가 목록표와 같은지
  디지털화  양자화 간격(정수 샘플 차이의 최대공약수) → 유효 비트 수, 최대 눈금에 닿은 비율(클리핑), DC
  스펙트럼  전체 파워의 5·50·95 % 누적 주파수, 200 Hz·1 kHz 아래 파워 비율
  사건 대역 주파수 칸마다 시간 방향의 P95 / P50 (dB) 을 구한다. 잡음뿐이면 이 값은 약 6.4 dB 다
            (한 창의 파워는 지수분포를 따르므로 ln20/ln2). 이보다 RISE_MARGIN_DB 넘게 솟은 만큼을 무게로 삼아
            그 무게가 모여 있는 주파수 범위 sig_f05, sig_f50, sig_f95 [Hz] 를 구한다.
            절대 파워로 재면 저주파 배경 잡음이 값을 차지하므로(돌고래 휘슬 컷의 전체 파워 중앙 주파수가 78 Hz 로 나온다),
            칸마다 자기 중앙값에 견준 상대값으로 잰다. '시간에 따라 변하는 성분이 어느 대역에 있는가'를 보는 값이다.
  활동      시간-주파수 칸이 자기 칸의 중앙값보다 TF_DB 넘게 솟은 비율(tf_active_frac. 잡음뿐이면 약 0.004),
            프레임의 광대역 레벨이 중앙값보다 6 dB 넘게 솟은 비율(active_frac), 그 P95-P50 [dB](rise_db)
  정상 선   클립 내내(시간의 90 % 이상) 주변보다 8 dB 넘게 솟아 있는 좁은 선의 수와 주파수

구현에서 고른 값: 창 길이 32 ms(샘플 수는 fs 에 맞춰 2의 거듭제곱으로), 50 % 겹침, 6 dB·8 dB·90 % 기준.
한 파일에서 앞 120초까지만 분석한다(wb_common.MAX_ANALYSIS_S). 더 긴 파일은 qc_truncated 로 표시된다. 헤더 대조는 파일 전체의 프레임 수로 한다.
문헌값이 아니라 이 자료를 훑어보려고 고른 값이므로, 결론에 쓰기 전에 연구자의 청취 결과와 맞춰 본다.

쓰는 법  python src/b04_scan_audio_qc.py [--set core|gold|all] [--limit N]
출력    work/qc/audio_qc_<set>.csv, work/qc/audio_qc_<set>_by_species.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from math import gcd

import numpy as np
import pandas as pd

from wb_common import DATA_DIR, MAX_ANALYSIS_S, index_audio, load_config, log, work_dir

WIN_S = 0.032          # 창 길이(초). 샘플 수는 fs 에 맞춘다
RISE_DB = 6.0          # 활동 프레임 기준
NOISE_RISE_DB = 10 * np.log10(np.log(20) / np.log(2))   # 잡음뿐일 때 P95/P50 = 6.36 dB
RISE_MARGIN_DB = 3.0   # 잡음 기대값보다 이만큼 넘게 솟은 칸만 '사건'으로 센다
TF_DB = 9.0            # 시간-주파수 칸의 활동 기준 (중앙값 대비)
LINE_DB = 8.0          # 정상 선 기준
LINE_TIME_FRAC = 0.90  # 정상 선이 유지되어야 하는 시간 비율


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


def cum_freq(power: np.ndarray, freqs: np.ndarray, qs=(0.05, 0.5, 0.95)) -> list[float]:
    tot = power.sum()
    if not np.isfinite(tot) or tot <= 0:
        return [np.nan] * len(qs)
    c = np.cumsum(power) / tot
    return [float(freqs[min(np.searchsorted(c, q), len(freqs) - 1)]) for q in qs]


def qc_one(path: str) -> dict:
    import soundfile as sf
    info = sf.info(path)
    n_total = int(info.frames)
    n_read = min(n_total, int(round(MAX_ANALYSIS_S * info.samplerate)))
    xi, fs = sf.read(path, frames=n_read, dtype="int16", always_2d=True)
    xi = xi[:, 0].astype(np.int64)
    n = len(xi)
    out = dict(file_fs=int(fs), file_frames=n_total, file_dur_s=round(n_total / fs, 4),
               qc_analyzed_s=round(n / fs, 4), qc_truncated=bool(n < n_total))
    if n < 64:
        out["qc_error"] = "too_short"
        return out

    # ---- 디지털화 -------------------------------------------------------------------
    d = np.unique(np.abs(np.diff(xi)))
    d = d[d > 0]
    step = 0
    for v in d[:2000]:
        step = gcd(step, int(v))
        if step == 1:
            break
    out["q_step"] = int(step) if step else 0
    out["eff_bits"] = round(16 - np.log2(step), 1) if step else np.nan
    peak = int(np.abs(xi).max())
    out["peak_abs"] = peak
    out["clip_frac"] = round(float((np.abs(xi) >= 32767 - (step or 1)).mean()), 6)
    out["dc_frac"] = round(float(xi.mean() / 32768.0), 5)
    x = xi.astype(np.float64) / 32768.0
    x = x - x.mean()
    rms = float(np.sqrt((x ** 2).mean()))
    out["rms_dbfs"] = round(20 * np.log10(rms + 1e-12), 2)

    # ---- 스펙트럼 -------------------------------------------------------------------
    nfft = nfft_for(fs, n)
    P = stft_power(x, nfft)                       # (T, F)
    freqs = np.fft.rfftfreq(nfft, 1 / fs)
    lo = freqs >= max(10.0, 2 * fs / nfft)        # DC 근처 칸은 뺀다
    mean_p = P.mean(0)
    f05, f50, f95 = cum_freq(mean_p[lo], freqs[lo])
    out.update(nfft=nfft, df_hz=round(fs / nfft, 3), n_frames_stft=int(P.shape[0]),
               f05_hz=round(f05, 1), f50_hz=round(f50, 1), f95_hz=round(f95, 1))
    tot = mean_p[lo].sum() + 1e-30
    out["p_lt200"] = round(float(mean_p[lo & (freqs < 200)].sum() / tot), 4)
    out["p_lt1k"] = round(float(mean_p[lo & (freqs < 1000)].sum() / tot), 4)

    # ---- 사건 대역: 시간 방향 P95 - P50 ------------------------------------------------
    if P.shape[0] >= 8:
        p50 = np.percentile(P, 50, axis=0)
        p95 = np.percentile(P, 95, axis=0)
        p10 = np.percentile(P, 10, axis=0)
        rise = 10 * np.log10((p95 + 1e-30) / (p50 + 1e-30))
        # 주파수 방향으로 3칸 이동 중앙값: 한 칸짜리 튐을 누른다
        rs = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(rise, 1, mode="edge"), 3), axis=1)
        wgt = np.clip(rs - (NOISE_RISE_DB + RISE_MARGIN_DB), 0, None)
        wgt[~lo] = 0
        if wgt.sum() > 0:
            e05, e50, e95 = cum_freq(wgt[lo], freqs[lo])
            wtot = wgt.sum()
            out.update(sig_f05_hz=round(e05, 1), sig_f50_hz=round(e50, 1), sig_f95_hz=round(e95, 1),
                       sig_w_lt200=round(float(wgt[freqs < 200].sum() / wtot), 4),
                       sig_w_lt1k=round(float(wgt[freqs < 1000].sum() / wtot), 4),
                       sig_bins_frac=round(float((wgt[lo] > 0).mean()), 4))
        else:
            out.update(sig_f05_hz=np.nan, sig_f50_hz=np.nan, sig_f95_hz=np.nan, sig_w_lt200=np.nan,
                       sig_w_lt1k=np.nan, sig_bins_frac=0.0)
        k = int(np.argmax(np.where(lo, rs, -np.inf)))
        out["rise_peak_hz"] = round(float(freqs[k]), 1)
        out["rise_peak_db"] = round(float(rs[k]), 2)
        out["tf_active_frac"] = round(float((P[:, lo] > p50[lo][None, :] * 10 ** (TF_DB / 10)).mean()), 5)
        # 활동: 프레임별 광대역 레벨
        lvl = 10 * np.log10(P[:, lo].sum(1) + 1e-30)
        med = np.median(lvl)
        out["active_frac"] = round(float((lvl > med + RISE_DB).mean()), 4)
        out["rise_db"] = round(float(np.percentile(lvl, 95) - med), 2)
        # 정상 선: P10 이 주변(주파수 방향 이동 중앙값)보다 LINE_DB 넘게 솟은 칸
        p10db = 10 * np.log10(p10 + 1e-30)
        w = max(9, (len(p10db) // 32) | 1)
        pad = np.pad(p10db, w // 2, mode="edge")
        base = np.median(np.lib.stride_tricks.sliding_window_view(pad, w), axis=1)
        strong = (P > (10 ** ((base + LINE_DB) / 10))[None, :]).mean(0) >= LINE_TIME_FRAC
        strong &= lo
        line_bins = np.flatnonzero(strong)
        # 이웃한 칸은 한 선으로 센다
        lines = []
        for b in line_bins:
            if lines and b - lines[-1][-1] <= 1:
                lines[-1].append(b)
            else:
                lines.append([b])
        centers = [float(freqs[int(np.mean(g))]) for g in lines]
        out["n_lines"] = len(centers)
        out["n_lines_ge10k"] = int(sum(c >= 10000 for c in centers))
        out["line_hz"] = ";".join(f"{c:.0f}" for c in centers[:8])
    else:
        out["qc_error"] = "few_frames"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="core", choices=["core", "gold", "all"], help="core=핵심 표본, gold=골드 목록, all=찾은 음원 전부")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    idx = index_audio(load_config()["audio_root"])
    if args.set == "core":
        sel = m[m.role == "core"]
    elif args.set == "gold":
        sel = m[m.gold != ""]
    else:
        sel = m
    sel = sel[sel.cut_id.isin(idx)]
    if args.limit:
        sel = sel.head(args.limit)
    log(f"대상 {len(sel)}개 (음원을 찾은 것만). 집합: {args.set}")
    rows, t0 = [], time.time()
    for i, r in enumerate(sel.itertuples(index=False)):
        try:
            q = qc_one(idx[r.cut_id])
        except Exception as e:  # noqa: BLE001
            q = dict(qc_error=f"{type(e).__name__}: {e}")
        q["cut_id"] = r.cut_id
        rows.append(q)
        if (i + 1) % 500 == 0:
            log(f"  {i + 1}/{len(sel)}  {time.time() - t0:.0f}초")
    qc = pd.DataFrame(rows)
    if "file_fs" not in qc:
        err = qc.qc_error.value_counts().to_dict() if "qc_error" in qc else {}
        log(f"음원을 하나도 읽지 못했다. 오류: {err}\n먼저 python src/b03_check_env.py 로 음원 폴더와 soundfile 을 확인한다.")
        return 1
    keep = ["cut_id", "species", "species_ko", "group3", "pool", "role", "split", "fs_hz", "n_frames", "fs_family",
            "g_tape", "g_site", "type_weak", "hold_reason"]
    out = sel[keep].merge(qc, on="cut_id", how="left")
    out["hdr_fs_ok"] = out.file_fs == out.fs_hz.astype(int)
    out["hdr_frames_ok"] = out.file_frames == out.n_frames.astype(int)
    # 검토 표시 (판정이 아니라 '먼저 볼 것')
    if "sig_f95_hz" in out:
        out["flag_event_lf"] = out.sig_f95_hz < 1000                      # 솟은 성분의 95 % 가 1 kHz 아래
        out["flag_quiet"] = (out.sig_bins_frac < 0.02) & (out.tf_active_frac < 0.008)   # 솟는 성분이 거의 없다
        out["flag_clip"] = out.clip_frac > 0.001
    qdir = work_dir("qc")
    out.to_csv(qdir / f"audio_qc_{args.set}.csv", index=False, encoding="utf-8-sig")

    spec = dict(
        n=("cut_id", "size"), tapes=("g_tape", "nunique"),
        fs_min=("file_fs", "min"), fs_max=("file_fs", "max"),
        bits_med=("eff_bits", "median"), clip_n=("clip_frac", lambda s: int((s > 0.001).sum())),
        f50_med=("f50_hz", "median"), sig_f05_med=("sig_f05_hz", "median"), sig_f50_med=("sig_f50_hz", "median"),
        sig_f95_med=("sig_f95_hz", "median"),
        event_lf_n=("flag_event_lf", "sum"), quiet_n=("flag_quiet", "sum"),
        rise_peak_db_med=("rise_peak_db", "median"), tf_active_med=("tf_active_frac", "median"),
        lines_ge10k_n=("n_lines_ge10k", lambda s: int((s > 0).sum())),
    )
    spec = {k: v for k, v in spec.items() if v[0] in out.columns}      # 계산되지 않은 열은 건너뛴다
    by = out.groupby(["group3", "species"]).agg(**spec).reset_index()
    by.to_csv(qdir / f"audio_qc_{args.set}_by_species.csv", index=False, encoding="utf-8-sig")
    log(by.to_string(index=False))
    log(f"\n헤더 fs 일치 {int(out.hdr_fs_ok.sum())}/{len(out)}, 프레임 수 일치 {int(out.hdr_frames_ok.sum())}/{len(out)}")
    if "qc_error" in out:
        log("오류:", out.qc_error.dropna().value_counts().to_dict())
    log(f"저장: {qdir / f'audio_qc_{args.set}.csv'}  ({time.time() - t0:.0f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
