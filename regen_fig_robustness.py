"""Regenerate only fig_robustness with ordinal x-positions to fix label overlap."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Publication style (same as make_figures.py)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "yellow": "#F0E442", "skyblue": "#56B4E9", "pink": "#CC79A7",
      "black": "#000000", "grey": "#999999"}
OUT = "figures_paper"

df = pd.read_csv(f"{OUT}/robustness_perseed.csv")
seeds = df["seed"].tolist()
seed_f1 = df["macro_f1"].tolist()

fig, ax = plt.subplots(figsize=(3.5, 2.4))
# Use ordinal x-positions so closely-valued seeds (7,13,42,99,2026) don't
# collapse on the axis; show seed value as the tick label instead.
xs = np.arange(len(seeds))
ax.scatter(xs, seed_f1, s=35, color=OI["blue"], zorder=3, label="fresh split + init")
ax.axhline(np.mean(seed_f1), color=OI["blue"], lw=0.9, ls="--")
ax.axhspan(np.mean(seed_f1) - np.std(seed_f1), np.mean(seed_f1) + np.std(seed_f1),
           color=OI["blue"], alpha=0.12)
ax.text(len(seeds) - 1, np.mean(seed_f1) + np.std(seed_f1) + 0.002,
        f"mean {np.mean(seed_f1):.3f} $\\pm$ {np.std(seed_f1):.3f}",
        ha="right", va="bottom", fontsize=7.2, color=OI["blue"])
ax.set_xticks(xs)
ax.set_xticklabels(seeds, rotation=0)
ax.set_xlim(-0.5, len(seeds) - 0.5)
ax.set_xlabel("Random seed (indexed)")
ax.set_ylabel("Macro-F1 (male test)")
ax.set_ylim(0.48, 0.58)
fig.tight_layout()

# Save both PDF and PNG
fig.savefig(f"{OUT}/fig_robustness.pdf")
fig.savefig(f"{OUT}/fig_robustness.png", dpi=300)
print(f"[saved] {OUT}/fig_robustness.pdf / .png")
print(f"seeds: {seeds}")
print(f"f1:    {seed_f1}")
print(f"mean={np.mean(seed_f1):.3f} std={np.std(seed_f1):.3f}")
