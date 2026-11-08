"""Figure for the signal-type-first MVP → docs/figures/fig8_mvp.png"""
import json, pathlib
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = json.loads((ROOT / "results/mvp_signal_type_first.json").read_text())
M = R["models"]
order = ["meta_fs", "band_only", "mvp", "mvp_shape", "flat_physical"]
label = {"meta_fs": "Sampling rate only", "band_only": "Band only", "mvp": "Signal type + band",
         "mvp_shape": "Signal type + band + shape", "flat_physical": "All physical (flat)"}
grey, dark, accent = "#9aa1a7", "#3a4146", "#e8a62a"
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.4, 1, 1]})

y = np.arange(len(order))[::-1]
for yi, k in zip(y, order):
    v = M[k]; c = accent if k.startswith("mvp") else grey
    lo, hi = v["ba_range_over_partitions"]
    ax[0].barh(yi, v["ba_mean_over_partitions"], color=c, height=0.6)
    ax[0].plot([lo, hi], [yi, yi], color=dark, lw=1.2)
    ax[0].text(hi + 0.012, yi, f"{v['ba_mean_over_partitions']:.2f}", va="center")
ax[0].axvline(0.797, color=dark, ls="--", lw=0.8); ax[0].text(0.80, y[0] + 0.45, "AST (manuscript)", fontsize=7, color=dark)
ax[0].axvline(1 / 3, color=grey, ls=":", lw=0.8)
ax[0].set_yticks(y); ax[0].set_yticklabels([label[k] for k in order]); ax[0].set_xlim(0.3, 0.9)
ax[0].set_xlabel("Taxon balanced accuracy\n(source-disjoint, mean and range of 20 partitions)")
ax[0].set_title("(a) Taxon classification", loc="left", fontsize=9)

S = R["fs40k_subset"]
vals = [S["ba"][k] for k in order]
ax[1].barh(y, vals, color=[accent if k.startswith("mvp") else grey for k in order], height=0.6)
for yi, v in zip(y, vals): ax[1].text(v + 0.01, yi, f"{v:.2f}", va="center")
ax[1].axvline(0.5, color=grey, ls=":", lw=0.8)
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].set_xlim(0.3, 0.9)
ax[1].set_xlabel("Balanced accuracy, Odontoceti vs Pinnipedia")
ax[1].set_title(f'(b) Sampling rate ≥ 40 kHz only (n = {S["n"]})', loc="left", fontsize=9)

cov = [1.0, 0.9, 0.8, 0.7, 0.5]
for k, c, ls in [("mvp_shape", accent, "-"), ("flat_physical", dark, "--"), ("meta_fs", grey, ":")]:
    ax[2].plot([100 * q for q in cov], [M[k]["coverage"][str(q)]["ba"] for q in cov], ls, color=c, marker="o", ms=3, label=label[k])
ax[2].invert_xaxis(); ax[2].set_xlabel("Clips classified automatically (%)\nrest routed to review")
ax[2].set_ylabel("Balanced accuracy"); ax[2].legend(frameon=False, fontsize=7)
ax[2].set_title("(c) Abstaining on low confidence (partition 1)", loc="left", fontsize=9)
fig.tight_layout()
fig.savefig(ROOT / "docs/figures/fig8_mvp.png", dpi=200, bbox_inches="tight")
