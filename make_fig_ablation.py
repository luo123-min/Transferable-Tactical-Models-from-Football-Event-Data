"""Regenerate fig_ablation.pdf/png with all 8 systems from ablation_8systems.csv.

Standalone (does NOT touch make_figures.py) so the published figure pipeline
is not disturbed; merge into make_figures.py later if desired.
"""
import pandas as pd, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("results_gnn/ablation_8systems.csv").sort_values("macro_f1")
labels = df["model"].tolist()
f1 = df["macro_f1"].values
acc = df["accuracy"].values
is_ours = ["ours" in m.lower() for m in labels]

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ypos = np.arange(len(df))
for i, (f1v, acv, ours) in enumerate(zip(f1, acc, is_ours)):
    c = "#C00000" if ours else "#222222"
    ax.plot([f1v, acv], [i, i], color=c, lw=1.2, alpha=0.8, zorder=2)
    ax.scatter(f1v, i, s=34, color=c, zorder=3, marker="o")
    ax.scatter(acv, i, s=24, facecolor="white", edgecolor=c, lw=1.0, zorder=3, marker="s")
    ax.text(f1v, i + 0.33, f"{f1v:.3f}", va="bottom", ha="center", fontsize=7.5,
            color=c, fontweight="bold" if ours else "normal")
    ax.text(acv, i + 0.33, f"{acv:.3f}", va="bottom", ha="center", fontsize=7.0,
            color="#888888")
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.0)
ax.set_xlabel("Macro-F1 (dots) / Accuracy (open squares)")
ax.set_xlim(0.10, 0.72)
ax.set_xticks([0.2, 0.3, 0.4, 0.5, 0.6])
ax.set_ylim(-0.6, len(df) - 0.1)
ax.axvline(1/3, color="grey", lw=0.7, ls=":", zorder=1)
ax.text(1/3 + 0.008, len(df) - 0.65, "chance (1/3)", fontsize=7, color="grey",
        ha="left", va="top")
ax.text(0.12, len(df) - 0.15, "ours 5-seed: 0.533 ± 0.012", ha="left", va="top",
        fontsize=6.5, color="#C00000")
ax.set_title("Eight-system ablation (male test split, seed 42)", fontsize=8.5)
fig.tight_layout()
fig.savefig("figures_paper/fig_ablation.pdf", dpi=300)
fig.savefig("figures_paper/fig_ablation.png", dpi=300)
print("saved figures_paper/fig_ablation.pdf/.png with", len(df), "systems")
