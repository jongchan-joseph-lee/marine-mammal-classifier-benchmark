"""Background timeline for the README / blog.  python 04_figures/fig_background.py -> docs/figures/fig0_background.png"""
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({"font.family": "serif", "font.serif": ["Liberation Serif", "DejaVu Serif"]})
INK, GREY, LIGHT, AMBER, AMBER_BG = "#333333", "#8a8a8a", "#f2f2f2", "#c47f12", "#fbeedb"
ERAS = [
    ("Classic PAM", "signal shape", [
        ("2000", "Spectrogram correlation", "Mellinger & Clark"),
        ("2016", "Watkins DB online", "Sayigh et al."),
        ("2017", "Sound types & bands", "Erbe et al.")]),
    ("Pretrained audio models", "transfer learning", [
        ("2017", "VGGish", "Hershey et al."),
        ("2021", "Spectrogram Transformer", "Gong et al."),
        ("2023", "Bioacoustic embeddings", "Ghani et al.")]),
    ("Evaluation issues", "generalisation", [
        ("2020", "Shortcut learning", "Geirhos et al."),
        ("2022", "Watkins in random splits", "BEANS, Hagiwara et al."),
        ("2022", "Test on novel conditions", "Stowell")]),
]
USED = ["signal type · frequency band\nphysical descriptors", "AST · VGGish · EfficientNet-B3\nMFCC + linear probe", "source-disjoint split\nmetadata baseline · abstention"]

fig, ax = plt.subplots(figsize=(14, 6.2))
ax.set_xlim(0, 14); ax.set_ylim(0, 6.2); ax.axis("off")
W, X0 = 4.3, [0.1, 4.85, 9.6]
for x, (title, span, items), used in zip(X0, ERAS, USED):
    ax.add_patch(FancyBboxPatch((x, 1.75), W, 4.2, boxstyle="round,pad=0,rounding_size=0.12", fc=LIGHT, ec=INK, lw=1.6))
    ax.text(x + W / 2, 5.55, title, ha="center", fontsize=16, fontweight="bold", color=INK)
    ax.text(x + W / 2, 5.15, span, ha="center", fontsize=13, color=GREY)
    for k, (yr, what, who) in enumerate(items):
        y = 4.45 - k * 0.9
        ax.text(x + 0.25, y, yr, fontsize=13.5, fontweight="bold", color=INK, va="center")
        ax.text(x + 1.05, y + 0.14, what, fontsize=13.5, color=INK, va="center")
        ax.text(x + 1.05, y - 0.22, who, fontsize=12, color=GREY, va="center", style="italic")
    ax.add_patch(FancyBboxPatch((x, 0.2), W, 1.05, boxstyle="round,pad=0,rounding_size=0.12", fc=AMBER_BG, ec=AMBER, lw=1.6))
    ax.text(x + W / 2, 1.0, "used in this project", ha="center", fontsize=12, color=AMBER, style="italic")
    ax.text(x + W / 2, 0.55, used, ha="center", va="center", fontsize=12.5, color=INK, linespacing=1.3)
    ax.annotate("", xy=(x + W / 2, 1.27), xytext=(x + W / 2, 1.73),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=1.5, mutation_scale=15))
for x in X0[:2]:
    ax.annotate("", xy=(x + W + 0.43, 3.85), xytext=(x + W + 0.02, 3.85),
                arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, mutation_scale=16))
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "figures", "fig0_background.png")
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
