"""Regenerate only fig_ablation as a dumbbell chart.

Each system is one horizontal line whose LEFT end is macro-F1 (filled dot)
and RIGHT end is accuracy (open square). The line length encodes the gap
between the two metrics: a short line means a healthy, consistent system
(ours), a long line shifted left means a collapsed system (GCN raw only).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
OI = {"verm": "#D55E00", "black": "#000000", "blue": "#0072B2", "grey": "#999999"}
OUT = "figures_paper"

abl = pd.read_csv(f"{OUT}/ablation_results.csv")
abl_s = abl.sort_values("macro_f1")           # ascending -> ours (max F1) at the top
labels = list(abl_s["model"])                  # one line, no wrapping
is_ours = ["ours" in m for m in abl_s["model"]]

fig, ax = plt.subplots(figsize=(6.0, 3.2))
ypos = np.arange(len(abl_s))                   # y=0 bottom (GCN raw) ... y=5 top (ours)
for i, (f1v, acv, ours) in enumerate(zip(abl_s["macro_f1"], abl_s["accuracy"], is_ours)):
    c = OI["verm"] if ours else OI["black"]
    # dumbbell stem: macro-F1 (left) -> accuracy (right)
    ax.plot([f1v, acv], [i, i], color=c, lw=1.2, alpha=0.8, zorder=2)
    ax.scatter(f1v, i, s=34, color=c, zorder=3, marker="o")           # macro-F1 dot
    ax.scatter(acv, i, s=24, facecolor="white", edgecolor=c, lw=1.0,
               zorder=3, marker="s")                                   # accuracy open square
    # value labels above each marker
    ax.text(f1v, i + 0.33, f"{f1v:.3f}", va="bottom", ha="center", fontsize=7.5,
            color=c, fontweight="bold" if ours else "normal")
    ax.text(acv, i + 0.33, f"{acv:.3f}", va="bottom", ha="center", fontsize=7.0,
            color=OI["grey"])
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=7.0)
ax.set_xlabel("Macro-F1 (dots) / Accuracy (open squares)")
ax.set_xlim(0.10, 0.72)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim(-0.6, len(abl_s) - 0.1)
# chance reference (macro-F1 chance = 1/3 for three classes)
ax.axvline(1/3, color="grey", lw=0.7, ls=":", zorder=1)
ax.text(1/3 + 0.008, 4.2, "chance", fontsize=7, color="grey",
        ha="left", va="top")
# 5-seed robustness note for ours (mean over fresh seeds, not the seed-42 run plotted)
ax.text(0.12, len(abl_s) - 0.15, "ours 5-seed: 0.533 ± 0.012",
        ha="left", va="top", fontsize=6.5, color=OI["verm"])
fig.tight_layout()

fig.savefig(f"{OUT}/fig_ablation.pdf")
fig.savefig(f"{OUT}/fig_ablation.png", dpi=300)
print(f"[saved] {OUT}/fig_ablation.pdf / .png")
