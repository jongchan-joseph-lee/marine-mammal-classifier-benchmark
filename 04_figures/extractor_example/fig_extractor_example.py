#!/usr/bin/env python3
"""Figure: how each representation turns one clip into a vector (8101403Z, North Atlantic right whale moan).

Input:  results/extractor_example_8101403Z.npz (from compute_example.py)
Output: docs/figures/extractor_example.png
Spectrogram-like inputs are drawn in greyscale (dark = high level); output vectors are the clip's
actual values, standardised within each vector (z-score) and drawn with a diverging red/blue map.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from scipy.signal import spectrogram

ROOT = Path(__file__).resolve().parents[2]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 7.5, "axes.linewidth": 0.5, "mathtext.fontset": "stix", "savefig.dpi": 300})
K = "#000000"
SPEC = "gray_r"
R = np.load(ROOT / "results/extractor_example_8101403Z.npz")
fs = int(R["fs"]); x = R["x"].astype(float)


def norm(A, lo=20, hi=99.7):
    l, h = np.percentile(A, [lo, hi])
    return np.clip((A - l) / (h - l + 1e-12), 0, 1)


def zgrid(v, shape):
    v = np.nan_to_num(np.asarray(v, float))
    z = (v - v.mean()) / (v.std() + 1e-12)
    out = np.full(int(np.prod(shape)), np.nan); out[:len(z)] = z
    return out.reshape(shape)


def clean(a):
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_linewidth(0.5)


rows = [
    dict(name="AST", pre="16 kHz, tiled to 10.24 s\nlog-mel 128 × 1024",
         net="1,212 patches (16 × 16)\nTransformer, 12 layers\npooled output",
         train="AudioSet", dim=768, shape=(24, 32), key="ast_vec", inp="ast"),
    dict(name="VGGish", pre="16 kHz\nlog-mel 64 × 96 per 0.96 s",
         net="4 conv blocks + 3 FC\n128 per 0.96 s patch\nmean over patches",
         train="AudioSet", dim=128, shape=(8, 16), key="vgg_vec", inp="vgg"),
    dict(name="EfficientNet-B3", pre="40 kHz, tiled to 5 s\nSTFT image 300 × 800",
         net="MBConv CNN\nglobal average pooling",
         train="ImageNet", dim=1536, shape=(32, 48), key="eff_vec", inp="eff"),
    dict(name="MFCC", pre="16 kHz, 25 ms frames\n64 mel bands, 50–8,000 Hz",
         net="log → DCT (c1–c20)\nmean, s.d., Δ mean, Δ s.d.",
         train="no training", dim=80, shape=(4, 20), key="mfcc_vec", inp="mfcc"),
    dict(name="Physical descriptors", pre="native 14.9 kHz\nSTFT, 32 ms window",
         net="band energy, temporal,\ntonal, impulsive and\nrepetition measures",
         train="no training", dim=51, shape=(3, 17), key="phys_vec", inp="phys"),
]

fig = plt.figure(figsize=(7.4, 6.3))
gs = fig.add_gridspec(len(rows), 6, width_ratios=[1.3, 1.3, 1.4, 1.45, 1.35, 0.42],
                      left=0.01, right=0.99, top=0.9, bottom=0.08, wspace=0.28, hspace=0.55)

# (a) original clip, spanning all rows
f, t, S = spectrogram(x - x.mean(), fs, nperseg=512, noverlap=448)
a0 = fig.add_subplot(gs[1:4, 0])
a0.imshow(10 * np.log10(S + 1e-20), origin="lower", aspect="auto", cmap=SPEC, extent=[t[0], t[-1], 0, f[-1] / 1000],
          vmin=np.percentile(10 * np.log10(S + 1e-20), 30), vmax=np.percentile(10 * np.log10(S + 1e-20), 99.8))
a0.set_xlabel("Time (s)"); a0.set_ylabel("Frequency (kHz)")
a0.tick_params(labelsize=6.5, length=2)
a0.set_title("Input clip", fontsize=8, fontweight="bold", pad=4)
a0.text(0, -0.36, "North Atlantic right whale moan\n8101403Z · 14.9 kHz · 2.16 s\nharmonics 0.4–6 kHz",
        transform=a0.transAxes, va="top", fontsize=6.8)

out_axes = []
for i, r in enumerate(rows):
    # representation + preprocessing
    a = fig.add_subplot(gs[i, 1]); a.axis("off")
    a.text(0, 0.98, r["name"], fontsize=8, fontweight="bold", va="top")
    a.text(0, 0.62, r["pre"], fontsize=6.5, va="top", linespacing=1.25)
    # model input
    b = fig.add_subplot(gs[i, 2]); clean(b)
    if r["inp"] == "ast":
        b.imshow(norm(R["ast_in"].T), origin="lower", aspect="auto", cmap=SPEC)
    elif r["inp"] == "vgg":
        V = np.concatenate(list(R["vgg_in"]), 0)
        b.imshow(norm(V.T), origin="lower", aspect="auto", cmap=SPEC)
        b.axvline(96, color=K, lw=0.6, ls=":")
    elif r["inp"] == "eff":
        b.imshow(R["eff_in"], aspect="auto", cmap="gray", vmin=0, vmax=1)
        b.text(0.5, 0.75, "no energy above 7.45 kHz\n(original Nyquist)", transform=b.transAxes, ha="center", va="center", fontsize=6, color="white")
    elif r["inp"] == "mfcc":
        c = R["mfcc_c"]; cz = (c - c.mean(0)) / (c.std(0) + 1e-9)
        b.imshow(cz.T, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    else:
        b.imshow(10 * np.log10(S + 1e-20), origin="lower", aspect="auto", cmap=SPEC,
                 vmin=np.percentile(10 * np.log10(S + 1e-20), 30), vmax=np.percentile(10 * np.log10(S + 1e-20), 99.8))
    # processing box
    c_ = fig.add_subplot(gs[i, 3]); c_.axis("off")
    c_.text(0.5, 0.5, r["net"], ha="center", va="center", fontsize=6.5, linespacing=1.3,
            bbox=dict(boxstyle="round,pad=0.45", fc="#f2f2f2", ec="#808080", lw=0.6))
    c_.text(0.5, -0.12, f"pretrained on {r['train']}" if r["train"] not in ("no training",) else "no training",
            ha="center", va="top", fontsize=6, style="italic", color="#555555")
    # output vector
    d = fig.add_subplot(gs[i, 4]); clean(d)
    im = d.imshow(zgrid(R[r["key"]], r["shape"]), aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5, interpolation="nearest")
    out_axes.append(d)
    # dim
    e = fig.add_subplot(gs[i, 5]); e.axis("off")
    e.text(0.5, 0.5, f"{r['dim']:,}", ha="center", va="center", fontsize=8.5, fontweight="bold")
    # arrows input -> processing -> output
    for src, dst in [(b, c_), (c_, d)]:
        fig.add_artist(ConnectionPatch((1.02, 0.5), (-0.02, 0.5), "axes fraction", "axes fraction", axesA=src, axesB=dst,
                                       arrowstyle="-|>", color="#555555", lw=0.6, mutation_scale=6))
    if i == 0:
        for ax_, h in [(a, "Representation"), (b, "Model input"), (c_, "Processing"), (d, "Output vector"), (e, "Dim.")]:
            ax_.set_title(h, fontsize=8, fontweight="bold", pad=10, loc="center")

pos = out_axes[-1].get_position()
cax = fig.add_axes([pos.x0, 0.035, pos.width, 0.011])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("value standardised within each vector (z)", fontsize=6.5); cb.ax.tick_params(labelsize=6, length=2)
fig.savefig(ROOT / "docs/figures/extractor_example.png", facecolor="white", bbox_inches="tight")
print("wrote docs/figures/extractor_example.png")
