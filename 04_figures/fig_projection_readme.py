#!/usr/bin/env python3
"""README projection figures for VGGish vectors: UMAP (3 panels) and t-SNE (2 panels).

Input: results/projection_vggish_core3539.npz — 2-D coordinates (uint16) and labels for the
3,539-clip sample, taken from the screening dashboard run (03_web, Watkins 스크리닝 결과집).
Same design as manuscript Figure 4 (29 fitted / 20 unseen recording sources) but a separate
run, so the agreement values differ slightly from the manuscript.
Agreement = balanced accuracy of a 15-nearest-neighbour vote in the 2-D map (build_maps.knn_ba).

Usage: python 04_figures/fig_projection_readme.py
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 9, "axes.linewidth": 0.5, "savefig.dpi": 300, "legend.frameon": False,
                     "mathtext.fontset": "stix"})
W = 6.3
PAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948",
       "#0f6e73", "#9a6324", "#9aa3e0", "#a3cc4e"]          # manuscript Figure 4 palette
OTHER = "#c9cfce"
TAXON = {"Odontoceti": ("Odontocetes", "#bdbdbd"), "Mysticeti": ("Mysticetes", "#1a1a1a"), "Pinnipedia": ("Pinnipeds", "#e34948")}

z = np.load(ROOT / "results/projection_vggish_core3539.npz")
score = json.loads(str(z["score_json"]))
sp, tx, split = z["species"], z["taxon"], z["split"]
top = pd.Series(sp).value_counts().index[:12].tolist()
col = np.array([OTHER] * len(sp), dtype=object)
for i, s in enumerate(top):
    col[sp == s] = PAL[i]
order = np.random.default_rng(0).permutation(len(sp))


def xy(key):
    return z[key].astype(float) / 65535


def frame(a, title, sub):
    a.set_xticks([]); a.set_yticks([])
    a.set_title(title, fontsize=8.5, loc="left")
    a.set_xlabel(sub, fontsize=8)
    for s in a.spines.values():
        s.set_visible(True); s.set_linewidth(0.5)


def species_legend(fig, y):
    n_other = len(set(sp)) - len(top)
    fig.legend([Line2D([], [], marker="o", color=c, ls="", ms=4) for c in PAL[:len(top)]] + [Line2D([], [], marker="o", color=OTHER, ls="", ms=4)],
               top + [f"{n_other} other species"], ncol=4, loc="upper center", bbox_to_anchor=(0.5, y),
               fontsize=7.5, columnspacing=1.0, handletextpad=0.2)


# UMAP: unsupervised, species-supervised, fitted on development sources + unseen sources projected
fig, ax = plt.subplots(1, 3, figsize=(W, 2.75))
P = xy("umap"); ax[0].scatter(P[order, 0], P[order, 1], s=2.5, c=list(col[order]), alpha=0.8, linewidths=0, rasterized=True)
frame(ax[0], "(a) No labels", f"species agreement {score['umap']['species']:.2f}")
P = xy("umap_sup"); ax[1].scatter(P[order, 0], P[order, 1], s=2.5, c=list(col[order]), alpha=0.8, linewidths=0, rasterized=True)
frame(ax[1], "(b) Species labels supplied", f"species agreement {score['umap_sup']['species']:.2f}")
P = xy("umap_sup_dev"); tr = np.where(split == "dev")[0]; te = np.where(split == "test")[0]
ax[2].scatter(P[tr, 0], P[tr, 1], s=2.5, c=list(col[tr]), alpha=0.2, linewidths=0, rasterized=True)
ax[2].scatter(P[te, 0], P[te, 1], s=5, c=list(col[te]), alpha=0.9, linewidths=0, rasterized=True)
n_tr, n_te = len(set(z["source"][tr])), len(set(z["source"][te]))
frame(ax[2], f"(c) Fitted on {n_tr} sources;\n      {n_te} unseen sources projected",
      f"fitted clips {score['umap_sup_dev']['dev_species']:.2f}; unseen sources {score['umap_sup_dev']['test_species']:.2f}")
species_legend(fig, 0.06)
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(OUT / "fig4_umap.png", bbox_inches="tight", facecolor="white"); plt.close(fig)

# t-SNE: unsupervised only (no label-supervised variant, no projection of unseen data)
P = xy("tsne")
fig, ax = plt.subplots(1, 2, figsize=(W * 0.8, 3.0))
ax[0].scatter(P[order, 0], P[order, 1], s=2.5, c=list(col[order]), alpha=0.8, linewidths=0, rasterized=True)
frame(ax[0], "(a) Coloured by species", f"species agreement {score['tsne']['species']:.2f}")
tcol = np.array([TAXON[t][1] for t in tx], dtype=object)
ax[1].scatter(P[order, 0], P[order, 1], s=2.5, c=list(tcol[order]), alpha=0.6, linewidths=0, rasterized=True)
frame(ax[1], "(b) Coloured by taxon", f"taxon agreement {score['tsne']['taxon']:.2f}")
ax[1].legend([Line2D([], [], marker="o", color=c, ls="", ms=4) for _, c in TAXON.values()], [n for n, _ in TAXON.values()],
             loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=3, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
species_legend(fig, 0.06)
fig.tight_layout(rect=(0, 0.1, 1, 1)); fig.savefig(OUT / "fig4b_tsne.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
print("wrote fig4_umap.png, fig4b_tsne.png")
