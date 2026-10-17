#!/usr/bin/env python3
"""figs.py — manuscript v3 figures (print-ready PNG, 300 dpi, text width 16 cm)."""
import json, base64, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"], "font.size": 9, "axes.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.dpi": 300, "figure.dpi": 100,
                     "legend.frameon": False, "mathtext.fontset": "stix"})
W = 6.3
S = json.load(open("/home/claude/summary/summary.json")); G = json.load(open("/home/claude/regroup/regroup3.json"))
REF = "/home/claude/ref_v1/eval/reference_v1"
FE = ["physv0", "mfcc16k", "vggish", "effb3", "ast_zero", "ast_tile"]
NM = {"meta": "Baseline (no audio)", "physv0": "Physical descriptors", "mfcc16k": "MFCC", "vggish": "VGGish", "effb3": "EfficientNet-B3", "ast_zero": "AST (zero-padded)", "ast_tile": "AST (repeated)"}
K = "0.15"; BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

# ── Figure 2: data structure ──
A = S["A"]; fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
ks = [k for k in A["site_hist"] if A["site_hist"][k] > 0]; v = [A["site_hist"][k] for k in ks]
b = ax[0].bar(ks, v, color=[K] + ["0.65"] * (len(ks) - 1), width=0.62)
for r, n in zip(b, v): ax[0].text(r.get_x() + r.get_width() / 2, n + 0.4, str(n), ha="center", va="bottom")
ax[0].set_xlabel("Recording sources per species"); ax[0].set_ylabel("Number of species"); ax[0].set_ylim(0, 21); ax[0].set_title("(a)", loc="left", fontsize=9)
shade = ["0.92", "0.78", "0.62", "0.45", "0.28", "0.08"]; names = {"이빨고래류": "Odontocetes", "수염고래류": "Mysticetes", "기각류": "Pinnipeds"}
for i, r in enumerate(A["fs"]["rows"][::-1]):
    x = 0
    for s, c in zip(r["share"], shade):
        ax[1].barh(i, s * 100, left=x, color=c, edgecolor="white", linewidth=0.8, height=0.6); x += s * 100
ax[1].set_yticks(range(3)); ax[1].set_yticklabels([f"{names[r['name']]}\n(n = {r['n']:,})" for r in A["fs"]["rows"][::-1]]); ax[1].set_xlim(0, 100)
ax[1].set_xlabel("Share of clips (%)"); ax[1].set_title("(b)", loc="left", fontsize=9); ax[1].spines["left"].set_visible(False); ax[1].tick_params(axis="y", length=0)
ax[1].legend([Patch(facecolor=c, edgecolor="0.5", linewidth=0.3) for c in shade], [f.replace("k", " kHz") for f in A["fs"]["fam"]], title="Sampling-rate family", ncol=6, loc="upper center",
             bbox_to_anchor=(0.5, -0.32), handlelength=1, handletextpad=0.4, columnspacing=0.9, fontsize=8, title_fontsize=8)
fig.tight_layout(); fig.savefig("fig/fig2.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 3: leakage and baseline ──
r = pd.read_csv(f"{REF}/results.csv"); r = r[(r.subset == "all") & (r.metric == "balanced_acc")]
fig, ax = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True); rows = ["meta"] + FE
for a, (task, title) in zip(ax, [("species", "Species (10 classes)"), ("type_single", "Call type (7 classes)"), ("group3", "Taxon (3 classes)")]):
    q = r[r.task == task]; base = q[(q.protocol == "allcv") & (q.feature == "meta")].value.iloc[0]; ch = q.chance.iloc[0]
    a.axvline(ch, color="0.5", ls=":", lw=0.8); a.axvline(base, color=K, ls="--", lw=0.8)
    for i, f in enumerate(rows):
        y = len(rows) - 1 - i; fc = q[(q.protocol == "filecv") & (q.feature == f)].iloc[0]; sc = q[(q.protocol == "allcv") & (q.feature == f)].iloc[0]
        a.plot([sc.value, fc.value], [y, y], color="0.75", lw=0.8, zorder=1)
        a.plot([sc.ci_lo, sc.ci_hi], [y, y], color=K, lw=1.3, zorder=2)
        a.plot(fc.value, y, "o", mfc="white", mec=K, ms=5, zorder=3); a.plot(sc.value, y, "D" if f == "meta" else "o", color=K, ms=5, zorder=4)
    a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_title(title, fontsize=9); a.set_xlabel("Balanced accuracy"); a.set_ylim(-0.6, len(rows) - 0.4)
ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([NM[f] for f in rows][::-1])
fig.legend([Line2D([], [], marker="o", mfc="white", mec=K, ls=""), Line2D([], [], marker="o", color=K, ls="-", lw=1.3), Line2D([], [], color=K, ls="--", lw=0.8), Line2D([], [], color="0.5", ls=":", lw=0.8)],
           ["random split of clips", "recording sources kept apart (95% CI)", "baseline, sources apart", "chance"], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=8, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("fig/fig3.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 4: UMAP with and without species labels (VGGish) ──
D = json.load(open("/home/claude/watkins_maps/real/data.json")); C = D["cuts"]
dec = lambda s, t: np.frombuffer(base64.b64decode(s), dtype=t)
blk = D["sup"]["vggish"]; rows_ = dec(blk["rows"], "<u2"); sp = np.array(C["sp"])[rows_]; spl = np.array(C["split"])[rows_]
top = pd.Series(sp).value_counts().index[:12].tolist()
pal = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#0f6e73", "#9a6324", "#9aa3e0", "#a3cc4e"]
col = np.array(["#c9cfce"] * len(sp), dtype=object)
for i, s in enumerate(top): col[sp == s] = pal[i]
EN = {"범고래": "Killer whale", "혹등고래": "Humpback whale", "참거두고래": "Long-finned pilot whale", "들쇠고래": "Short-finned pilot whale", "참돌고래": "Common dolphin", "큰돌고래": "Bottlenose dolphin",
      "북극고래": "Bowhead whale", "흰고래": "Beluga", "일각고래": "Narwhal", "큰머리돌고래": "Risso's dolphin", "긴부리돌고래": "Spinner dolphin", "대서양알락돌고래": "Atlantic spotted dolphin",
      "알락돌고래": "Pantropical spotted dolphin", "줄박이돌고래": "Striped dolphin", "대서양낫돌고래": "Atlantic white-sided dolphin", "흑범고래": "False killer whale", "북대서양긴수염고래": "North Atlantic right whale",
      "웨델물범": "Weddell seal", "클리멘돌고래": "Clymene dolphin", "남방긴수염고래": "Southern right whale", "프레이저돌고래": "Fraser's dolphin", "턱수염물범": "Bearded seal", "뱀머리돌고래": "Rough-toothed dolphin", "흰부리돌고래": "White-beaked dolphin", "사라와크돌고래": "Fraser's dolphin"}
M = blk["metrics"]; fig, ax = plt.subplots(1, 3, figsize=(W, 2.75))
panels = [("unsup", "(a) No labels", f"species agreement {M['unsup']['all']:.2f}"), ("sup_all", "(b) Species labels supplied", f"species agreement {M['sup_all']['all']:.2f}"),
          ("sup_holdout", "(c) Fitted on 29 sources;\n      20 unseen sources projected", f"fitted clips {M['sup_holdout']['train']:.2f}; unseen sources {M['sup_holdout']['heldout']:.2f}")]
order = np.random.default_rng(0).permutation(len(sp))
for a, (k, t, sub) in zip(ax, panels):
    xy = dec(blk[k], "<u2").reshape(-1, 2) / 65535
    if k == "sup_holdout":
        tr = spl == 1; a.scatter(xy[tr, 0], xy[tr, 1], s=2.5, c=list(col[tr]), alpha=0.25, linewidths=0, rasterized=True)
        te = np.where(spl == 2)[0]; a.scatter(xy[te, 0], xy[te, 1], s=7, c=list(col[te]), edgecolors="black", linewidths=0.35, rasterized=True)
    else: a.scatter(xy[order, 0], xy[order, 1], s=2.5, c=list(col[order]), alpha=0.8, linewidths=0, rasterized=True)
    a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=8.5, loc="left"); a.set_xlabel(sub, fontsize=8)
    for s_ in a.spines.values(): s_.set_visible(True); s_.set_linewidth(0.5)
names_ = [D["cats"]["sp"][s] for s in top]
missing = [n for n in names_ if n not in EN]; print("species without English name:", missing)
fig.legend([Line2D([], [], marker="o", color=c, ls="", ms=4) for c in pal[:len(top)]] + [Line2D([], [], marker="o", color="#c9cfce", ls="", ms=4)], [EN.get(n, n) for n in names_] + ["24 other species"],
           ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.06), fontsize=7.5, columnspacing=1.0, handletextpad=0.2)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig/fig4.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 5: three signal types ──
fig, ax = plt.subplots(1, 4, figsize=(W, 2.6), sharey=True); rows = ["meta"] + FE
for a, (k, t) in zip(ax, [("3_random", "(a) Random split"), ("3_site", "(b) Sources apart,\n      all clips"), ("3_site_fs40k", "(c) Sources apart,\n      40-kHz family"), ("3_site_fs80k", "(d) Sources apart,\n      80-kHz family")]):
    s = G[k]; a.axvline(1 / 3, color="0.5", ls=":", lw=0.8); a.axvline(s["feat"]["meta"]["ba"], color=K, ls="--", lw=0.8)
    for i, f in enumerate(rows):
        y = len(rows) - 1 - i; v = s["feat"][f]; a.plot([v["lo"], v["hi"]], [y, y], color=K, lw=1.1)
        if f == "meta": a.plot(v["ba"], y, "D", color=K, ms=4.5)
        else: a.plot(v["ba"], y, "o", mfc=K if v["dlo"] > 0 else "white", mec=K, ms=5)
    a.set_xlim(0.15, 1); a.set_xticks([0.2, 0.6, 1.0]); a.set_title(t, fontsize=8.5, loc="left"); a.set_ylim(-0.6, len(rows) - 0.4)
    a.set_xlabel(f"Balanced accuracy\n{s['n']:,} clips, {s['sites']} sources", fontsize=8)
ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([NM[f] for f in rows][::-1])
fig.legend([Line2D([], [], marker="o", color=K, ls=""), Line2D([], [], marker="o", mfc="white", mec=K, ls=""), Line2D([], [], marker="D", color=K, ls="--", lw=0.8), Line2D([], [], color="0.5", ls=":", lw=0.8)],
           ["exceeds baseline (CI of paired difference > 0)", "does not", "baseline", "chance"], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=8, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("fig/fig5.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 6: single descriptors ──
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3)); sty = [(K, "-"), ("0.55", "-"), (K, ":")]
spec = [("(a) Tonal concentration", None, None, {"tonal": "tonal", "impulsive": "impulsive", "pulsed": "pulsed"}),
        ("(b) Centre frequency of the signal band", [2, 3, 4, 5], ["0.1", "1", "10", "100 kHz"], {"whistle": "whistle", "moan": "moan"}),
        ("(c) Inter-click interval", [-3, -2, -1, 0], ["1", "10", "100", "1000 ms"], {"click": "click", "pulse_train": "pulse train"})]
for a, d, (t, tk, tl, lab) in zip(ax, S["D"], spec):
    e = np.array(d["bins"])
    for g, (c, ls) in zip(d["groups"], sty):
        h = np.array(g["h"]); a.stairs(h, e, color=c, ls=ls, lw=1.2, label=f"{lab[g['name']]} (n = {g['n']})", fill=(ls == "-" and c != K), alpha=1 if not (ls == "-" and c != K) else 0.35)
        if ls == "-" and c != K: a.stairs(h, e, color=c, lw=1.0)
    if tk: a.set_xticks(tk); a.set_xticklabels(tl)
    a.set_yticks([]); a.spines["left"].set_visible(False); a.set_title(t, fontsize=8.5, loc="left"); a.set_xlabel(f"AUROC {d['auroc']:.2f}", fontsize=8)
    a.set_ylim(0, a.get_ylim()[1] * 1.45); a.legend(fontsize=7.5, loc="upper right", handlelength=1.4)
fig.tight_layout(); fig.savefig("fig/fig6.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 7: signal type × band → taxon ──
E = S["E"]; fig, a = plt.subplots(figsize=(W, 3.0)); cols = [K, "0.6", "white"]; hat = [None, None, "////"]; y = 0; yt, yl = [], []
bandn = {"1 kHz 미만": "< 1 kHz", "1–8 kHz": "1–8 kHz", "8 kHz 초과": "> 8 kHz"}
for ty in ["pulsed", "impulsive", "tonal"]:
    for bnd in ["8 kHz 초과", "1–8 kHz", "1 kHz 미만"]:
        c = next(c for c in E["cells"] if c["type"] == ty and c["band"] == bnd); x = 0
        if bnd == "1–8 kHz": a.axhspan(y - 0.46, y + 0.46, color="#fff1c2", zorder=0)
        if c["n"] < 20:
            a.barh(y, 100, color="white", edgecolor="0.6", linewidth=0.5, linestyle="--", height=0.62); a.text(50, y, "fewer than 20 clips", ha="center", va="center", fontsize=7.5, color="0.4")
        else:
            for s, fc, h in zip(c["share"], cols, hat):
                a.barh(y, s * 100, left=x, color=fc, edgecolor=K, linewidth=0.5, hatch=h, height=0.62)
                if s > 0.08: a.text(x + s * 50, y, f"{s*100:.0f}", ha="center", va="center", fontsize=7.5, color="white" if fc == K else "black", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none") if h else None)
                x += s * 100
        a.text(102, y, f"n = {c['n']}", va="center", fontsize=8); yt.append(y); yl.append(bandn[bnd]); y += 1
    a.text(-17, y - 2, ty, va="center", ha="right", fontsize=9, fontweight="bold"); y += 0.5
a.set_yticks(yt); a.set_yticklabels(yl); a.set_xlim(0, 100); a.set_xlabel("Share of clips (%)"); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
a.legend([Patch(facecolor=c, edgecolor=K, hatch=h, linewidth=0.5) for c, h in zip(cols, hat)], ["Odontocetes", "Mysticetes", "Pinnipeds"], ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), fontsize=8)
fig.tight_layout(); fig.savefig("fig/fig7.png", bbox_inches="tight"); plt.close(fig)
print("done")
