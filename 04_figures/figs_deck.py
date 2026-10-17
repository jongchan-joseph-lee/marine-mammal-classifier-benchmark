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
NM = {"meta": "Metadata baseline", "physv0": "Physical descriptors", "mfcc16k": "MFCC", "vggish": "VGGish", "effb3": "EfficientNet-B3", "ast_zero": "AST (zero-padded)", "ast_tile": "AST (repeated)"}
K = "#0B4EA2"; BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"

# ── Figure 2: data structure ──
A = S["A"]; fig, ax = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"width_ratios": [1, 1.5]})
ks = [k for k in A["site_hist"] if A["site_hist"][k] > 0]; v = [A["site_hist"][k] for k in ks]
b = ax[0].bar(ks, v, color=[K] + ["#7FCDEB"] * (len(ks) - 1), width=0.62)
for r, n in zip(b, v): ax[0].text(r.get_x() + r.get_width() / 2, n + 0.4, str(n), ha="center", va="bottom")
ax[0].set_xlabel("Recording sources per species"); ax[0].set_ylabel("Number of species"); ax[0].set_ylim(0, 21); ax[0].set_title("(a)", loc="left", fontsize=9)
shade = ["#E3F4FB", "#B9E3F5", "#7FCDEB", "#33B5E5", "#1A7FC4", "#0B4EA2"]; names = {"이빨고래류": "Odontocetes", "수염고래류": "Mysticetes", "기각류": "Pinnipeds"}
for i, r in enumerate(A["fs"]["rows"][::-1]):
    x = 0
    for s, c in zip(r["share"], shade):
        ax[1].barh(i, s * 100, left=x, color=c, edgecolor="white", linewidth=0.8, height=0.6); x += s * 100
ax[1].set_yticks(range(3)); ax[1].set_yticklabels([f"{names[r['name']]}\n(n = {r['n']:,})" for r in A["fs"]["rows"][::-1]]); ax[1].set_xlim(0, 100)
ax[1].set_xlabel("Share of clips (%)"); ax[1].set_title("(b)", loc="left", fontsize=9); ax[1].spines["left"].set_visible(False); ax[1].tick_params(axis="y", length=0)
ax[1].legend([Patch(facecolor=c, edgecolor="0.5", linewidth=0.3) for c in shade], [f.replace("k", " kHz") for f in A["fs"]["fam"]], title="Sample-rate family", ncol=6, loc="upper center",
             bbox_to_anchor=(0.5, -0.32), handlelength=1, handletextpad=0.4, columnspacing=0.9, fontsize=8, title_fontsize=8)
fig.tight_layout(); fig.savefig("fig_deck/fig2.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 3: leakage and baseline ──
r = pd.read_csv(f"{REF}/results.csv"); r = r[(r.subset == "all") & (r.metric == "balanced_acc")]
fig, ax = plt.subplots(1, 3, figsize=(W, 2.9), sharey=True); rows = ["meta"] + FE
for a, (task, title) in zip(ax, [("species", "Species (10 classes)"), ("type_single", "Call type (7 classes)"), ("group3", "Taxon (3 classes)")]):
    q = r[r.task == task]; base = q[(q.protocol == "allcv") & (q.feature == "meta")].value.iloc[0]; ch = q.chance.iloc[0]
    a.axvline(ch, color="0.5", ls=":", lw=0.8); a.axvline(base, color=K, ls="--", lw=0.8)
    for i, f in enumerate(rows):
        y = len(rows) - 1 - i; fc = q[(q.protocol == "filecv") & (q.feature == f)].iloc[0]; sc = q[(q.protocol == "allcv") & (q.feature == f)].iloc[0]
        a.plot([sc.value, fc.value], [y, y], color="#B9E3F5", lw=1.0, zorder=1)
        a.plot([sc.ci_lo, sc.ci_hi], [y, y], color=K, lw=1.3, zorder=2)
        a.plot(fc.value, y, "o", mfc="white", mec=K, ms=5, zorder=3); a.plot(sc.value, y, "D" if f == "meta" else "o", color=K, ms=5, zorder=4)
    a.set_xlim(0, 1); a.set_xticks([0, 0.5, 1]); a.set_title(title, fontsize=9); a.set_xlabel("Balanced accuracy"); a.set_ylim(-0.6, len(rows) - 0.4)
ax[0].set_yticks(range(len(rows))); ax[0].set_yticklabels([NM[f] for f in rows][::-1])
fig.legend([Line2D([], [], marker="o", mfc="white", mec=K, ls=""), Line2D([], [], marker="o", color=K, ls="-", lw=1.3), Line2D([], [], color=K, ls="--", lw=0.8), Line2D([], [], color="0.5", ls=":", lw=0.8)],
           ["random split of clips", "recording sources kept apart (95% CI)", "metadata baseline, sources apart", "chance"], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=8, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig("fig_deck/fig3.png", bbox_inches="tight"); plt.close(fig)

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
M = blk["metrics"]; fig, ax = plt.subplots(1, 4, figsize=(8.4, 2.75))
panels = [("unsup", "(a) No labels, all sources", f"species agreement {M['unsup']['all']:.2f}"), ("sup_all", "(b) Species labels, all sources", f"species agreement {M['sup_all']['all']:.2f}"),
          ("fit", "(c) Species labels, 29 sources", f"species agreement {M['sup_holdout']['train']:.2f}"), ("proj", "(d) 20 unseen sources projected", f"species agreement {M['sup_holdout']['heldout']:.2f}")]
order = np.random.default_rng(0).permutation(len(sp)); tr = np.where(spl == 1)[0]; te = np.where(spl == 2)[0]
for a, (k, t, sub) in zip(ax, panels):
    xy = dec(blk["sup_holdout" if k in ("fit", "proj") else k], "<u2").reshape(-1, 2) / 65535
    if k == "fit": o = tr[np.random.default_rng(0).permutation(len(tr))]; a.scatter(xy[o, 0], xy[o, 1], s=2.5, c=list(col[o]), alpha=0.8, linewidths=0, rasterized=True)
    elif k == "proj":
        a.scatter(xy[tr, 0], xy[tr, 1], s=1.5, c="#EEF1F3", linewidths=0, rasterized=True)
        a.scatter(xy[te, 0], xy[te, 1], s=6, c=list(col[te]), linewidths=0, alpha=0.9, rasterized=True)
    else: a.scatter(xy[order, 0], xy[order, 1], s=2.5, c=list(col[order]), alpha=0.8, linewidths=0, rasterized=True)
    a.set_xticks([]); a.set_yticks([]); a.set_title(t, fontsize=8.5, loc="left"); a.set_xlabel(sub, fontsize=8)
    for s_ in a.spines.values(): s_.set_visible(True); s_.set_linewidth(0.5)
ax[3].text(0.02, 0.02, "pale grey: clips of the 29 fitted sources", transform=ax[3].transAxes, fontsize=7, color="0.45")
names_ = [D["cats"]["sp"][s] for s in top]
missing = [n for n in names_ if n not in EN]; print("species without English name:", missing)
fig.legend([Line2D([], [], marker="o", color=c, ls="", ms=4) for c in pal[:len(top)]] + [Line2D([], [], marker="o", color="#c9cfce", ls="", ms=4)], [EN.get(n, n) for n in names_] + ["24 other species"],
           ncol=7, loc="upper center", bbox_to_anchor=(0.5, 0.06), fontsize=7.5, columnspacing=1.0, handletextpad=0.2)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig("fig_deck/fig4.png", bbox_inches="tight"); plt.close(fig)

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
           ["exceeds baseline (CI of paired difference > 0)", "does not", "metadata baseline", "chance"], ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.03), fontsize=8, columnspacing=1.2, handletextpad=0.5)
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig("fig_deck/fig5.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 6: single descriptors ──
fig, ax = plt.subplots(1, 3, figsize=(W, 2.3)); sty = [(K, "-"), ("#00B5E2", "-"), (K, ":")]
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
fig.tight_layout(); fig.savefig("fig_deck/fig6.png", bbox_inches="tight"); plt.close(fig)

# ── Figure 7: signal type × band → taxon, with icons and taxon pies ──
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
E = S["E"]; cols = [K, "#00B5E2", "white"]; hat = [None, None, "////"]; TX = ["Odontocetes", "Mysticetes", "Pinnipeds"]
fig = plt.figure(figsize=(7.8, 2.95)); gs = fig.add_gridspec(2, 2, width_ratios=[3.3, 1], height_ratios=[1, 1], wspace=0.12, hspace=0.35)
a = fig.add_subplot(gs[:, 0]); y = 0; yt, yl = [], []
SUB = {"tonal": "whistle · moan", "impulsive": "click · pulse train", "pulsed": "pulsed call · growl"}
bandn = {"1 kHz 미만": "< 1 kHz", "1–8 kHz": "1–8 kHz", "8 kHz 초과": "> 8 kHz"}; fig_tax = np.zeros(3); type_n = {}
for ty in ["pulsed", "impulsive", "tonal"]:
    tn = 0
    for bnd in ["8 kHz 초과", "1–8 kHz", "1 kHz 미만"]:
        c = next(c for c in E["cells"] if c["type"] == ty and c["band"] == bnd); x = 0; tn += c["n"]; fig_tax += np.round(np.array(c["share"]) * c["n"])
        if bnd == "1–8 kHz": a.axhspan(y - 0.46, y + 0.46, color="#DFF3FB", zorder=0)
        if c["n"] < 20:
            a.barh(y, 100, color="white", edgecolor="0.6", linewidth=0.5, linestyle="--", height=0.62); a.text(50, y, "fewer than 20 clips", ha="center", va="center", fontsize=7.5, color="0.4")
        else:
            for sh, fc, h in zip(c["share"], cols, hat):
                a.barh(y, sh * 100, left=x, color=fc, edgecolor=K, linewidth=0.5, hatch=h, height=0.62)
                if sh > 0.08: a.text(x + sh * 50, y, f"{sh*100:.0f}", ha="center", va="center", fontsize=7.5, color="white" if fc == K else "black", bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none") if h else None)
                x += sh * 100
        a.text(102, y, f"n = {c['n']}", va="center", fontsize=8); yt.append(y); yl.append(bandn[bnd]); y += 1
    type_n[ty] = tn
    a.text(-17, y - 2 + 0.5, ty, va="center", ha="right", fontsize=9.5, fontweight="bold")
    a.text(-17, y - 2 - 0.05, SUB[ty], va="center", ha="right", fontsize=7.5, color="0.5")
    a.text(-17, y - 2 - 0.55, f"n = {tn}", va="center", ha="right", fontsize=7, color="0.5"); y += 0.5
a.set_yticks(yt); a.set_yticklabels(yl); a.set_xlim(0, 100); a.set_xlabel("Share of clips (%)"); a.spines["left"].set_visible(False); a.tick_params(axis="y", length=0)
a.set_ylim(-0.6, y + 0.9)
# icon legend above the bars
for k, (f, lab, fc, h) in enumerate(zip(["ic_odo.png", "ic_mys.png", "ic_pin.png"], TX, cols, hat)):
    xc = 12 + k * 33; img = mpimg.imread("fig_deck/" + f)
    a.add_artist(AnnotationBbox(OffsetImage(img, zoom=0.062), (xc, y + 0.35), frameon=False, annotation_clip=False))
    a.add_patch(plt.Rectangle((xc + 5, y + 0.12), 3.2, 0.42, facecolor=fc, edgecolor=K, hatch=h, linewidth=0.5, clip_on=False))
    a.text(xc + 9.5, y + 0.33, lab, va="center", fontsize=8.5)
coh = np.array([8820, 1385, 489]); fig_tax = fig_tax.astype(int)
for r, (vals, title) in enumerate([(coh, f"Analysis cohort\n{coh.sum():,} clips"), (fig_tax, f"This figure\n{fig_tax.sum():,} clips")]):
    ax_ = fig.add_subplot(gs[r, 1]); w, _ = ax_.pie(vals, colors=cols, startangle=90, counterclock=False, wedgeprops=dict(edgecolor=K, linewidth=0.5))
    w[2].set_hatch("////")
    ax_.set_title(title, fontsize=8, pad=2)
    ax_.legend(w, [f"{v:,} ({v/vals.sum()*100:.0f}%)" for v in vals], loc="center left", bbox_to_anchor=(0.95, 0.5), fontsize=7.5, handlelength=1, handletextpad=0.4)
print("fig7 taxon counts", fig_tax, type_n)
fig.savefig("fig_deck/fig7.png", bbox_inches="tight"); plt.close(fig)
print("done")
