# -*- coding: utf-8 -*-
"""9단계. 듣고 보고 표기하는 화면을 만든다. 서버 없이 더블클릭으로 열린다.

쓰임 1  골드 표기: 연구자가 시험 해역의 컷을 직접 확인해 형태·품질 라벨을 남긴다 (기본값).
쓰임 2  실패 사례 확인: 평가에서 틀린 컷 목록(--list)을 같은 화면으로 연다.

만드는 것  <out>/index.html, <out>/img/<컷>.png, <out>/audio/<컷>.wav
  그림    파형 + 전체 대역 스펙트로그램 + 신호 대역 확대. 축은 초와 Hz, 원래 fs 로 계산한다.
          창 길이는 초로 정하고 샘플 수는 fs 에 맞춘다 (fs 계열별 값은 SPEC_WIN_S).
  소리    청취용 사본. 원본 샘플을 그대로 두고 WAV 헤더의 재생 속도만 바꾼다.
          fs 가 48 kHz 보다 높으면 느리게(음높이가 내려간다), 8 kHz 보다 낮으면 빠르게(올라간다) 재생된다.
          화면에 배속을 표시한다. 원본 폴더에는 아무것도 쓰지 않는다.
  표기    노트와 노트에서 읽은 형태는 처음에는 가려 둔다(먼저 보면 판정이 끌려간다). 버튼으로 열 수 있고, 열었는지가 기록된다.
          표기는 브라우저에 자동 저장되고 CSV 로 내보내거나 불러올 수 있다. 내보낸 CSV 가 정본이다.

쓰는 법  python src/b09_make_review_page.py [--set gold_practice|gold_test|gold] [--list 목록.csv] [--out 폴더] [--limit N]
  연습 30컷           python src/b09_make_review_page.py --set gold_practice
  시험 앞 120컷       python src/b09_make_review_page.py --set gold_test --limit 120
                      (골드 목록은 분류군을 돌아가며 뽑은 순서라서 앞 120컷은 분류군마다 40컷이다)
"""
from __future__ import annotations

import argparse
import html
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from wb_common import DATA_DIR, PKG_ROOT, TYPE_CLASSES, TYPE_KO, VERSION, index_audio, load_config, log

STORE_KEY = "Watkins_Bench_v1"      # 브라우저 저장소의 열쇠. 골드 목록이 v1 과 같으므로 판이 올라가도 바꾸지 않는다 (표기가 이어진다)

SPEC_WIN_S = [(5000, 0.512), (14142, 0.064), (28284, 0.032), (1e12, 0.0128)]   # (fs 상한, 창 길이 초)
MAX_FRAMES = 1600


def spec_params(fs: int, n: int) -> tuple[int, int]:
    win_s = next(w for lim, w in SPEC_WIN_S if fs < lim)
    nfft = int(2 ** round(np.log2(max(fs * win_s, 64))))
    while nfft > 64 and n < 2 * nfft:
        nfft //= 2
    hop = max(1, nfft // 8)
    if (n - nfft) // hop + 1 > MAX_FRAMES:
        hop = max(1, (n - nfft) // MAX_FRAMES)
    return nfft, hop


def spectrogram(x: np.ndarray, fs: int):
    nfft, hop = spec_params(fs, len(x))
    if len(x) < nfft:
        x = np.pad(x, (0, nfft - len(x)))
    nfr = 1 + (len(x) - nfft) // hop
    idx = np.arange(nfft)[None, :] + hop * np.arange(nfr)[:, None]
    S = np.abs(np.fft.rfft(x[idx] * np.hanning(nfft)[None, :], axis=1)) ** 2
    f = np.fft.rfftfreq(nfft, 1 / fs)
    t = (np.arange(nfr) * hop + nfft / 2) / fs
    return 10 * np.log10(S.T + 1e-20), f, t, nfft, hop


def signal_band(SdB: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """칸마다 P95-P50 이 큰 대역(누적 5-95 %). 확대 그림의 범위를 정하는 데만 쓴다."""
    if SdB.shape[1] < 8:
        return float(f[1]), float(f[-1])
    rise = np.percentile(SdB, 95, axis=1) - np.percentile(SdB, 50, axis=1)
    w = np.clip(rise - 9.4, 0, None)
    w[f < max(10.0, 2 * f[1])] = 0
    if w.sum() <= 0:
        return float(f[1]), float(f[-1])
    c = np.cumsum(w) / w.sum()
    return float(f[np.searchsorted(c, 0.05)]), float(f[min(np.searchsorted(c, 0.95), len(f) - 1)])


def listening_rate(fs: int) -> tuple[int, float]:
    """청취용 사본의 헤더 재생 속도와 배속(1 보다 작으면 느리게)."""
    rate = float(fs)
    while rate > 48000:
        rate /= 2
    while rate < 8000:
        rate *= 2
    return int(round(rate)), rate / fs


def render(path: str, cut_id: str, out_img: Path, out_wav: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import soundfile as sf

    xi, fs = sf.read(path, dtype="int16", always_2d=True)
    xi = xi[:, 0]
    x = xi.astype(np.float64) / 32768.0
    x = x - x.mean()
    SdB, f, t, nfft, hop = spectrogram(x, fs)
    lo, hi = signal_band(SdB, f)
    vmax = np.percentile(SdB, 99.9)
    vmin = max(np.percentile(SdB, 5), vmax - 70)
    dur = len(x) / fs

    z_lo = max(0.0, lo / 2)
    z_hi = min(fs / 2, hi * 1.5)
    zoom = (z_hi - z_lo) < 0.6 * (fs / 2)
    fig, axes = plt.subplots(3 if zoom else 2, 1, figsize=(9.6, 7.2 if zoom else 5.2),
                             gridspec_kw={"height_ratios": [1, 3, 3] if zoom else [1, 3]}, constrained_layout=True)
    ax = axes[0]
    tt = np.arange(len(x)) / fs
    step = max(1, len(x) // 20000)
    ax.plot(tt[::step], x[::step], lw=0.5, color="#1f4e5f")
    ax.set_xlim(0, dur)
    ax.set_ylabel("진폭")
    ax.set_title(f"{cut_id}   fs {fs:,} Hz   길이 {dur:.2f} s   창 {nfft / fs * 1000:.1f} ms (칸 {fs / nfft:.1f} Hz)", fontsize=10)
    ext = [0, dur, f[0], f[-1]]
    axes[1].imshow(SdB, origin="lower", aspect="auto", extent=ext, vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest")
    axes[1].set_ylabel("주파수 [Hz]  (전체 대역)")
    if zoom:
        axes[2].imshow(SdB, origin="lower", aspect="auto", extent=ext, vmin=vmin, vmax=vmax, cmap="magma", interpolation="nearest")
        axes[2].set_ylim(z_lo, z_hi)
        axes[2].set_ylabel("주파수 [Hz]  (신호 대역 확대)")
    axes[-1].set_xlabel("시간 [s]")
    for a in axes[1:]:
        a.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.savefig(out_img, dpi=80)
    plt.close(fig)

    rate, speed = listening_rate(fs)
    peak = max(1, int(np.abs(xi).max()))
    gain = min(8.0, 0.9 * 32767 / peak)                      # 작은 신호는 키운다 (사본만. 원본은 그대로)
    y = np.clip(np.round((xi - int(round(xi.mean()))) * gain), -32768, 32767).astype("<i2")
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(y.tobytes())
    return dict(fs=int(fs), dur=round(dur, 3), play_rate=rate, speed=round(speed, 4), gain=round(gain, 2),
                band_lo=round(lo, 1), band_hi=round(hi, 1), nfft=nfft)


def _font_setup() -> None:
    """한글 글꼴이 있으면 쓴다. 없으면 축 이름이 네모로 나올 수 있으나 판정에는 지장이 없다."""
    import matplotlib
    from matplotlib import font_manager
    names = {f.name for f in font_manager.fontManager.ttflist}
    for cand in ["Malgun Gothic", "NanumGothic", "AppleGothic", "Noto Sans CJK KR", "Noto Sans KR", "NanumBarunGothic"]:
        if cand in names:
            matplotlib.rcParams["font.family"] = cand
            break
    matplotlib.rcParams["axes.unicode_minus"] = False


PAGE = r"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f4f6f5;--card:#fff;--ink:#17262b;--mute:#5b6b70;--line:#d5dcda;--acc:#0e6b6b;--acc2:#b4530a;--ok:#1d7a46;--no:#a23b3b}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 "Malgun Gothic","Apple SD Gothic Neo","Noto Sans KR",system-ui,sans-serif;background:var(--bg);color:var(--ink)}
header{background:#12343b;color:#fff;padding:14px 20px}header h1{margin:0;font-size:19px}header p{margin:4px 0 0;color:#c9d8d6;font-size:13px}
.bar{display:flex;flex-wrap:wrap;gap:8px;align-items:center;padding:10px 20px;background:#e8eeec;border-bottom:1px solid var(--line)}
.bar input[type=text]{padding:6px 8px;border:1px solid var(--line);border-radius:6px;width:150px}
button,select{font:inherit;padding:6px 10px;border:1px solid var(--line);border-radius:6px;background:#fff;cursor:pointer}
button.pri{background:var(--acc);color:#fff;border-color:var(--acc)}
.wrap{display:grid;grid-template-columns:250px 1fr;gap:14px;padding:14px 20px}
@media(max-width:860px){.wrap{grid-template-columns:1fr}#list{max-height:200px}}
#list{background:var(--card);border:1px solid var(--line);border-radius:8px;overflow:auto;max-height:calc(100vh - 170px)}
.it{padding:7px 10px;border-bottom:1px solid #eef1f0;cursor:pointer;display:flex;justify-content:space-between;gap:6px;font-size:13px}
.it.sel{background:#dcefee}.it .st{font-size:12px;color:var(--mute)}.it.done .st{color:var(--ok);font-weight:600}.it.ex .st{color:var(--no)}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:12px}
.card h2{margin:0 0 6px;font-size:17px}.kv{color:var(--mute);font-size:13px}
img.spec{max-width:100%;border:1px solid var(--line);border-radius:6px;background:#000}
fieldset{border:1px solid var(--line);border-radius:8px;margin:0 0 10px;padding:8px 12px}legend{font-weight:600;font-size:14px;padding:0 4px}
label.c{display:inline-flex;align-items:center;gap:5px;margin:3px 12px 3px 0;white-space:nowrap}
textarea{width:100%;min-height:54px;font:inherit;padding:6px;border:1px solid var(--line);border-radius:6px}
.meta{background:#fbf6ec;border:1px dashed #d9b77a;border-radius:6px;padding:8px 10px;font-size:13px;display:none}
table.g{border-collapse:collapse;margin:8px 0;font-size:13.5px}table.g th{text-align:left;white-space:nowrap;padding:3px 12px 3px 0;vertical-align:top}table.g td{padding:3px 0}
details.card summary{cursor:pointer}
.hint{color:var(--mute);font-size:12.5px}.warn{color:var(--acc2)}kbd{border:1px solid var(--line);border-radius:4px;padding:0 4px;background:#fff;font-size:12px}
</style></head><body>
<header><h1>__TITLE__</h1><p>__SUB__</p></header>
<div class="bar">
  <select id="fStatus"><option value="">모든 상태</option><option value="todo">미표기</option><option value="done">표기함</option></select>
  <input type="text" id="fText" placeholder="컷 번호 검색">
  <span id="prog" class="kv"></span><span style="flex:1"></span>
  <label class="kv">표기자 <input type="text" id="reviewer" style="width:110px"></label>
  <button class="pri" id="btnExport">표기 CSV 저장</button>
  <label class="kv">불러오기 <input type="file" id="fileImport" accept=".csv"></label>
</div>
<div class="wrap">
  <div id="list"></div>
  <div>
    <div class="card">
      <div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">
        <button id="prev">← 이전 <kbd>A</kbd></button><button id="next">다음 → <kbd>D</kbd></button>
        <h2 id="ttl" style="margin:0 8px"></h2><span id="pos" class="kv"></span>
      </div>
      <p class="kv" id="info"></p>
      <audio id="au" controls preload="auto" style="width:100%"></audio>
      <p class="hint"><span id="spd"></span> &nbsp; 더 느리게 듣기 <select id="rate"><option value="1">그대로</option><option value="0.5">1/2</option><option value="0.25">1/4</option></select> <span class="hint">(음높이도 함께 내려간다. <kbd>Space</kbd> 재생·멈춤)</span></p>
      <img class="spec" id="im" alt="파형과 스펙트로그램">
      <p><button id="btnMeta">메타데이터 보기 (노트·노트에서 읽은 형태·종)</button> <span class="hint">표기를 먼저 하고 여는 것을 권한다. 열었는지가 기록된다.</span></p>
      <div class="meta" id="meta"></div>
    </div>
    <details class="card" id="guide"><summary><b>표기 기준</b> <span class="hint">(범주를 나누는 기준은 소리의 모양이다. 주파수 대역과 종은 기준이 아니다)</span></summary>
      <table class="g">
        <tr><th>클릭</th><td>낱개로 구분되는 충격음. 스펙트로그램에서 세로선이 하나씩 떨어져 보인다. 클릭이 여러 개 이어져도 낱개가 구분되면 클릭이다.</td></tr>
        <tr><th>펄스열·버스트</th><td>클릭·펄스가 빠르게 이어져 한 덩어리로 들린다(버즈, 버스트 펄스, creak). 세로선이 촘촘히 붙어 띠처럼 보이고, 거칠게 들린다.</td></tr>
        <tr><th>휘슬·스윕</th><td>가는 선(음조)이고 주파수가 뚜렷이 오르내린다. 대역은 따지지 않는다. 수염고래·물범의 낮은 스윕(upsweep, downsweep)도 여기에 넣는다.</td></tr>
        <tr><th>모운</th><td>가는 선(음조)이고 길며 주파수 변화가 작다.</td></tr>
        <tr><th>그롤·그런트</th><td>음조의 선이 뚜렷하지 않은 거칠고 낮은 소리.</td></tr>
        <tr><th>바크</th><td>짧고 거친 소리가 끊어져 되풀이된다.</td></tr>
        <tr><th>펄스성 호출·스퀼</th><td>펄스가 더 빨라 가로 방향의 촘촘한 배음 줄무늬가 뚜렷하고, 음높이가 있는 소리로 들린다(범고래의 call, squeal, scream). 펄스열과 가르기 어려우면 둘 다 고른다.</td></tr>
        <tr><th>노래</th><td>여러 단위가 일정한 차례로 이어진다. 단위 하나만 들어 있으면 그 단위의 모양으로 적고 메모에 '노래의 일부'라고 쓴다.</td></tr>
      </table>
      <p class="hint">헷갈리면 가까운 것을 모두 고르고 확신도를 낮춘다. <b>4. 물리적 모양</b>은 범주 이름과 따로 적는다. 나중에 범주의 경계를 바꿔도(예: 낮은 스윕을 모운으로) 이 칸으로 다시 나눌 수 있다.</p>
      <p class="hint"><b>이 기준은 초안이다.</b> 연습 30컷에서 기준이 맞지 않는 곳이 보이면 메모에 적는다. 시험 컷을 표기하기 전에 기준을 고친다.</p>
    </details>
    <div class="card" id="form">
      <fieldset><legend>1. 대상 동물의 소리가 있는가</legend>
        <label class="c"><input type="radio" name="present" value="y">있다</label>
        <label class="c"><input type="radio" name="present" value="n">없다</label>
        <label class="c"><input type="radio" name="present" value="unsure">모르겠다</label></fieldset>
      <fieldset><legend>2. 들어 있는 형태 (모두 고름) <span class="hint">숫자 키 1-9</span></legend><div id="types"></div></fieldset>
      <fieldset><legend>3. 대표 형태 (하나)</legend><div id="dom"></div></fieldset>
      <fieldset><legend>4. 물리적 모양 (모두 고름)</legend>
        <label class="c"><input type="checkbox" name="attr" value="tonal_contour">음조 윤곽(주파수가 선으로 보임)</label>
        <label class="c"><input type="checkbox" name="attr" value="fm_sweep">주파수가 뚜렷이 오르내림(스윕)</label>
        <label class="c"><input type="checkbox" name="attr" value="harmonic">배음 구조</label>
        <label class="c"><input type="checkbox" name="attr" value="single_click">낱개의 클릭</label>
        <label class="c"><input type="checkbox" name="attr" value="click_train">클릭열·버스트</label>
        <label class="c"><input type="checkbox" name="attr" value="broadband_rough">광대역의 거친 소리</label>
        <label class="c"><input type="checkbox" name="attr" value="lf_lt1k">1 kHz 아래가 중심</label></fieldset>
      <fieldset><legend>5. 방해음 (모두 고름)</legend>
        <label class="c"><input type="checkbox" name="intf" value="ship">선박·기계</label>
        <label class="c"><input type="checkbox" name="intf" value="hum_line">전원 험·가는 선</label>
        <label class="c"><input type="checkbox" name="intf" value="tape">테이프 잡음·히스</label>
        <label class="c"><input type="checkbox" name="intf" value="other_species">다른 종</label>
        <label class="c"><input type="checkbox" name="intf" value="water">수면·물소리·부딪힘</label>
        <label class="c"><input type="checkbox" name="intf" value="overload">과부하·찌그러짐</label>
        <label class="c"><input type="checkbox" name="intf" value="band_cut">대역이 잘린 듯함</label></fieldset>
      <fieldset><legend>6. 평가에 써도 되는가 / 확신도</legend>
        <label class="c"><input type="radio" name="usable" value="y">쓴다</label>
        <label class="c"><input type="radio" name="usable" value="n">뺀다</label>
        <label class="c"><input type="radio" name="usable" value="hold">보류</label>
        <span style="margin-left:18px">확신도</span>
        <label class="c"><input type="radio" name="conf" value="3">높음</label>
        <label class="c"><input type="radio" name="conf" value="2">보통</label>
        <label class="c"><input type="radio" name="conf" value="1">낮음</label></fieldset>
      <textarea id="memo" placeholder="메모 (판정 이유, 헷갈린 점)"></textarea>
      <p class="hint">표기는 이 브라우저에 자동 저장된다. 브라우저 저장은 지워질 수 있으니 끝날 때마다 <b>표기 CSV 저장</b>을 누른다.</p>
    </div>
  </div>
</div>
<script>
const DATA = __DATA__;
const TYPES = __TYPES__;
const KEY = "watkins_review__" + __KEYID__;
let ann = {}; try { ann = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e) { ann = {}; }
let cur = 0, view = DATA.map((_, i) => i);
const $ = id => document.getElementById(id);
function persist(){ try { localStorage.setItem(KEY, JSON.stringify(ann)); } catch(e) {} }
function isDone(a){ return a && (a.usable || a.present || (a.types && a.types.length)); }
function buildInputs(){
  $("types").innerHTML = TYPES.map((t,i) => `<label class="c"><input type="checkbox" name="type" value="${t[0]}"><kbd>${i+1}</kbd> ${t[1]}</label>`).join("") + `<label class="c"><input type="checkbox" name="type" value="other"><kbd>9</kbd> 그 밖의 것</label>`;
  $("dom").innerHTML = TYPES.map(t => `<label class="c"><input type="radio" name="dom" value="${t[0]}">${t[1]}</label>`).join("") + `<label class="c"><input type="radio" name="dom" value="other">그 밖의 것</label><label class="c"><input type="radio" name="dom" value="none">없음·모름</label>`;
}
function renderList(){
  const fs = $("fStatus").value, ft = $("fText").value.trim().toLowerCase();
  view = DATA.map((_, i) => i).filter(i => {
    const d = DATA[i], a = ann[d.id];
    if (fs === "todo" && isDone(a)) return false;
    if (fs === "done" && !isDone(a)) return false;
    if (ft && !d.id.toLowerCase().includes(ft)) return false;
    return true; });
  $("list").innerHTML = view.map(i => { const d = DATA[i], a = ann[d.id];
    const st = isDone(a) ? (a.usable === "n" ? "뺌" : "표기함") : "미표기";
    return `<div class="it ${i===cur?"sel":""} ${isDone(a)?"done":""} ${a&&a.usable==="n"?"ex":""}" data-i="${i}"><span>${d.set} ${d.order} · ${d.id}</span><span class="st">${st}</span></div>`; }).join("");
  const n = DATA.filter(d => isDone(ann[d.id])).length;
  $("prog").textContent = `표기 ${n} / ${DATA.length}`;
}
function setRadio(name, v){ document.querySelectorAll(`input[name=${name}]`).forEach(e => e.checked = (e.value === v)); }
function setChecks(name, vs){ document.querySelectorAll(`input[name=${name}]`).forEach(e => e.checked = (vs||[]).includes(e.value)); }
function show(i){
  cur = i; const d = DATA[i], a = ann[d.id] || {};
  $("ttl").textContent = d.id; $("pos").textContent = `${d.order} / ${DATA.length}  ·  ${d.set}`;
  $("info").textContent = `fs ${d.fs.toLocaleString()} Hz · 길이 ${d.dur} s · 솟은 성분의 대역(자동 추정) ${Math.round(d.band_lo).toLocaleString()}–${Math.round(d.band_hi).toLocaleString()} Hz`;
  $("im").src = "img/" + d.id + ".png"; $("au").src = "audio/" + d.id + ".wav"; applyRate();
  $("spd").innerHTML = d.speed === 1 ? `청취용 사본: 원래 속도 (재생 ${d.play_rate.toLocaleString()} Hz), 음량 ${d.gain}배`
    : `<span class="warn">청취용 사본: ${d.speed < 1 ? "느리게" : "빠르게"} ${d.speed < 1 ? "1/" + Math.round(1/d.speed) : Math.round(d.speed) + "배"} 재생 → 들리는 음높이도 같은 비율로 ${d.speed < 1 ? "내려간다" : "올라간다"}.</span> 그림의 축은 원래 Hz·초다. 음량 ${d.gain}배`;
  $("meta").style.display = a.meta_seen ? "block" : "none";
  const ex = [["범주를 정하지 못한 낱말", d.unplaced], ["동물의 소리가 아니라고 읽은 낱말", d.nonanimal], ["다른 종의 소리", d.other_sp], ["출처를 모르는 충격음", d.unknown], ["표시", d.flags], ["손으로 고친 이유", d.override]].filter(x => x[1]).map(x => `${x[0]}: ${x[1]}`).join(" · ");
  $("meta").innerHTML = `<b>${d.species_ko}</b> <i>${d.species}</i> · ${d.group_ko}<br>장소: ${d.location || "(없음)"} · 연도: ${d.year || "(없음)"}<br>노트: ${d.note || "(없음)"}<br>` +
    `<span class="hint">아래는 노트를 프로그램이 읽은 것이다. 사람이 확인한 라벨이 아니다.</span><br>노트에서 읽은 형태: <b>${d.type_weak || "(없음)"}</b> · 단일 형태 과제의 라벨: ${d.type_single || "(없음)"} · 낱말: ${d.terms || "(없음)"}` +
    (ex ? `<br>${ex}` : "") + `<br>v1 의 낱말 태그: ${d.type_tags || "(없음)"} (원 태그 ${d.tags_raw || "없음"})`;
  setRadio("present", a.present); setChecks("type", a.types); setRadio("dom", a.dominant); setChecks("attr", a.attrs);
  setChecks("intf", a.intf); setRadio("usable", a.usable); setRadio("conf", a.conf); $("memo").value = a.memo || "";
  renderList(); const el = document.querySelector(".it.sel"); if (el) el.scrollIntoView({block:"nearest"});
}
function collect(){
  const d = DATA[cur], a = ann[d.id] || {};
  const r = n => (document.querySelector(`input[name=${n}]:checked`) || {}).value || "";
  const c = n => [...document.querySelectorAll(`input[name=${n}]:checked`)].map(e => e.value);
  Object.assign(a, {present:r("present"), types:c("type"), dominant:r("dom"), attrs:c("attr"), intf:c("intf"), usable:r("usable"), conf:r("conf"), memo:$("memo").value, reviewer:$("reviewer").value, at:new Date().toISOString()});
  ann[d.id] = a; persist(); renderList();
}
function applyRate(){ const au = $("au"); try { au.preservesPitch = false; au.mozPreservesPitch = false; au.webkitPreservesPitch = false; } catch(e) {} au.defaultPlaybackRate = +$("rate").value; au.playbackRate = +$("rate").value; }
function step(k){ const p = view.indexOf(cur); const q = view[Math.min(view.length-1, Math.max(0, (p<0?0:p)+k))]; if (q !== undefined) show(q); }
function csvCell(s){ s = String(s == null ? "" : s); return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g,'""') + '"' : s; }
const COLS = ["cut_id","set","order","target_present","types","dominant_type","attrs","interference","usable","confidence","memo","meta_seen","reviewer","reviewed_at"];
function exportCsv(){
  const rows = [COLS.join(",")];
  DATA.forEach(d => { const a = ann[d.id]; if (!isDone(a)) return;
    rows.push([d.id, d.set, d.order, a.present, (a.types||[]).join(";"), a.dominant, (a.attrs||[]).join(";"), (a.intf||[]).join(";"), a.usable, a.conf, a.memo, a.meta_seen ? "y" : "n", a.reviewer, a.at].map(csvCell).join(",")); });
  const blob = new Blob(["﻿" + rows.join("\r\n") + "\r\n"], {type:"text/csv;charset=utf-8"});
  const u = URL.createObjectURL(blob), l = document.createElement("a");
  l.href = u; l.download = "gold_labels_" + new Date().toISOString().slice(0,10) + ".csv"; document.body.appendChild(l); l.click(); l.remove(); URL.revokeObjectURL(u);
}
function parseCsv(text){
  const out = []; let row = [], cell = "", q = false; text = text.replace(/^﻿/, "");
  for (let i = 0; i < text.length; i++){ const ch = text[i];
    if (q){ if (ch === '"'){ if (text[i+1] === '"'){ cell += '"'; i++; } else q = false; } else cell += ch; }
    else if (ch === '"') q = true; else if (ch === ","){ row.push(cell); cell = ""; }
    else if (ch === "\n"){ row.push(cell); out.push(row); row = []; cell = ""; } else if (ch !== "\r") cell += ch; }
  if (cell !== "" || row.length){ row.push(cell); out.push(row); } return out;
}
function importCsv(file){
  const fr = new FileReader(); fr.onload = () => { const rows = parseCsv(fr.result); const h = rows.shift(); const ix = Object.fromEntries(h.map((c,i) => [c,i]));
    if (ix.cut_id === undefined){ window.console.log("cut_id 열이 없는 파일"); return; }
    const known = new Set(DATA.map(d => d.id)); let n = 0;
    rows.forEach(r => { const id = r[ix.cut_id]; if (!known.has(id)) return; const g = k => (ix[k] === undefined ? "" : r[ix[k]] || "");
      ann[id] = {present:g("target_present"), types:g("types").split(";").filter(Boolean), dominant:g("dominant_type"), attrs:g("attrs").split(";").filter(Boolean), intf:g("interference").split(";").filter(Boolean), usable:g("usable"), conf:g("confidence"), memo:g("memo"), meta_seen:g("meta_seen")==="y", reviewer:g("reviewer"), at:g("reviewed_at")}; n++; });
    persist(); show(cur); $("prog").textContent += `  (불러옴 ${n}행)`; }; fr.readAsText(file, "utf-8");
}
buildInputs();
$("list").addEventListener("click", e => { const it = e.target.closest(".it"); if (it){ show(+it.dataset.i); } });
$("form").addEventListener("change", collect); $("memo").addEventListener("input", collect);
$("prev").onclick = () => step(-1); $("next").onclick = () => step(1);
$("fStatus").onchange = renderList; $("fText").oninput = renderList;
$("rate").onchange = applyRate;
$("btnExport").onclick = exportCsv; $("fileImport").onchange = e => { if (e.target.files[0]) importCsv(e.target.files[0]); };
$("btnMeta").onclick = () => { const d = DATA[cur]; const a = ann[d.id] || {}; a.meta_seen = true; ann[d.id] = a; persist(); $("meta").style.display = "block"; };
document.addEventListener("keydown", e => { if (["INPUT","TEXTAREA","SELECT"].includes(e.target.tagName) && e.target.type !== "checkbox" && e.target.type !== "radio") return;
  if (e.key === "d" || e.key === "ArrowRight") step(1); else if (e.key === "a" || e.key === "ArrowLeft") step(-1);
  else if (e.key === " "){ e.preventDefault(); const au = $("au"); au.paused ? au.play() : au.pause(); }
  else if (/^[1-9]$/.test(e.key)){ const boxes = document.querySelectorAll("input[name=type]"); const b = boxes[+e.key - 1]; if (b){ b.checked = !b.checked; collect(); } } });
show(0);
</script></body></html>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="gold", choices=["gold", "gold_practice", "gold_test"])
    ap.add_argument("--list", default="", help="cut_id 열이 있는 CSV. 주면 --set 대신 이 목록을 쓴다 (실패 사례 확인용)")
    ap.add_argument("--out", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--title", default="")
    args = ap.parse_args()

    m = pd.read_csv(DATA_DIR / "bench_manifest.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    if args.list:
        lst = pd.read_csv(args.list, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        ids = list(dict.fromkeys(lst.cut_id))
        sel = m.set_index("cut_id").loc[[c for c in ids if c in set(m.cut_id)]].reset_index()
        sel["set_name"] = Path(args.list).stem
        sel["order"] = range(1, len(sel) + 1)
        name = Path(args.list).stem
    else:
        g = m[m.gold != ""].copy()
        if args.set == "gold_practice":
            g = g[g.gold == "practice"]
        elif args.set == "gold_test":
            g = g[g.gold == "test"]
        g["order"] = g.gold_order.astype(int)
        g["set_name"] = g.gold.map({"practice": "연습", "test": "시험"})
        sel = g.sort_values(["gold", "order"])                 # practice 가 먼저 온다
        name = args.set
    if args.limit:
        sel = sel.head(args.limit)          # 목록의 앞 N컷. 음원이 있는지를 보기 전에 자른다 (앞 120컷의 구성이 바뀌지 않게)
    audio_root = load_config()["audio_root"]
    idx = index_audio(audio_root)
    missing = [c for c in sel.cut_id if c not in idx]
    sel = sel[sel.cut_id.isin(idx)]
    if len(sel) == 0:
        log(f"음원을 하나도 찾지 못했다 (대상 {len(missing)}컷). 음원 폴더를 확인한다: {audio_root}\n"
            "폴더가 다르면 watkins_bench_config.json 의 audio_root 를 고친다. 화면은 만들지 않았다.")
        return 1
    out = Path(args.out) if args.out else PKG_ROOT / "review" / name
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    log(f"대상 {len(sel)}개 (음원을 찾지 못한 것 {len(missing)}개) → {out}")
    _font_setup()

    data, t0 = [], time.time()
    for i, r in enumerate(sel.itertuples(index=False)):
        try:
            info = render(idx[r.cut_id], r.cut_id, out / "img" / f"{r.cut_id}.png", out / "audio" / f"{r.cut_id}.wav")
        except Exception as e:  # noqa: BLE001
            log(f"  그리기 실패 {r.cut_id}: {type(e).__name__}: {e}")
            continue
        esc = html.escape
        data.append(dict(id=r.cut_id, order=int(r.order), set=r.set_name, species=r.species, species_ko=r.species_ko,
                         group_ko=r.group3_ko, type_weak=r.type_weak, type_single=r.type_single, terms=esc(r.type_terms),
                         unplaced=esc(r.type_unplaced), nonanimal=esc(r.note_nonanimal), other_sp=esc(r.note_other_species),
                         unknown=esc(r.note_unknown), flags=esc(r.note_flags), override=esc(r.note_override),
                         type_tags=r.type_tags, tags_raw=r.call_tags_raw,
                         location=esc(r.location), year=r.year, note=esc(r.note), **info))
        if (i + 1) % 50 == 0:
            log(f"  {i + 1}/{len(sel)}  {time.time() - t0:.0f}초")
    title = args.title or ("Watkins 골드 표기" if not args.list else f"Watkins 사례 확인: {name}")
    sub = f"{VERSION} · {len(data)}컷 · 그림과 소리는 원래 fs 의 음원에서 만들었다 · 표기는 CSV 로 저장한다"
    page = (PAGE.replace("__TITLE__", html.escape(title)).replace("__SUB__", html.escape(sub))
            .replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
            .replace("__TYPES__", json.dumps([[t, TYPE_KO[t]] for t in TYPE_CLASSES], ensure_ascii=False))
            .replace("__KEYID__", json.dumps(f"{STORE_KEY}_{name}")))
    (out / "index.html").write_text(page, encoding="utf-8")
    if missing:
        (out / "missing_audio.txt").write_text("\n".join(missing), encoding="utf-8")
        log(f"주의: 음원을 찾지 못한 컷 {len(missing)}개는 화면에 없다. 목록: {out / 'missing_audio.txt'}")
    elif (out / "missing_audio.txt").exists():
        (out / "missing_audio.txt").unlink()                   # 앞선 실행이 남긴 목록 (이번에는 모두 찾았다)
    log(f"끝. {out / 'index.html'}  ({time.time() - t0:.0f}초)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
