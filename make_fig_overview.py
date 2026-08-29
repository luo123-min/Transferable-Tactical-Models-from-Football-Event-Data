"""
make_fig_overview.py — Figure 1: overall pipeline / conceptual overview.

Draws the study architecture: StatsBomb data source -> three analysis lines
(E1 clustering, E2 freeze-frame complementarity, E3 GCN outcome prediction)
-> cross-domain validation (gender + league transfer).

Style matches make_figures.py (Okabe-Ito palette, DejaVu Sans, pdf.fonttype=42,
300 dpi). Outputs figures_paper/fig_overview.pdf (vector) + .png (300 dpi).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# Shared style is re-declared here so the script can also run standalone.
OUT = "figures_paper"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
      "yellow": "#F0E442", "black": "#000000", "grey": "#666666"}
FILL = {"src": "#F5F5F5",
        "e1": "#FDF3E3", "e2": "#E2F3EE", "e3": "#E4F0F7", "val": "#FAEEF5"}
EDGE = {"e1": OI["orange"], "e2": OI["green"], "e3": OI["blue"], "val": OI["purple"]}


def box(ax, cx, cy, w, h, fc, ec, lw=0.9):
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.02,rounding_size=0.08",
                       linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2)
    ax.add_patch(p)


def arrow(ax, x1, y1, x2, y2, color="0.35", lw=0.9):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                shrinkA=0, shrinkB=0),
                zorder=1)


def txt(ax, x, y, s, size=8.0, color="black", weight="normal",
        style="normal", ha="center", va="center"):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight,
            fontstyle=style, ha=ha, va=va, zorder=3)


def draw_overview(out_dir=OUT):
    """Draw Figure 1 and save PDF+PNG."""
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # ---------------------------------------------------------------- layout
    # E1/E2/E3 three columns shifted DOWN by COL_DY; the cross-domain
    # validation box shifted UP by VAL_DY. All arrows/text track these.
    COL_DY = -0.30
    VAL_DY = +0.30
    yT, yM, yB = 6.30 + COL_DY, 5.00 + COL_DY, 3.60 + COL_DY   # 6.00, 4.70, 3.30
    val_cy = 1.225 + VAL_DY                                     # 1.525
    h1 = 0.88
    half = h1 / 2                                               # 0.44

    # ---------------------------------------------------------------- data source
    box(ax, 5.5, 7.50, 5.2, 0.90, FILL["src"], OI["black"], lw=1.0)
    txt(ax, 5.5, 7.71, "StatsBomb Open Data", size=10.5, weight="bold")
    txt(ax, 5.5, 7.42, "3,961 matches · 24 competitions · men's & women's football",
        size=8.5)
    txt(ax, 5.5, 7.18, "event streams  +  360 freeze-frames", size=8.5)

    # arrows from the data source down to the three lines
    for xs, cx in [(3.9, 2.0), (5.5, 5.5), (7.1, 9.0)]:
        arrow(ax, xs, 7.05, cx, yT + half)

    # column centres and box geometry
    cx1, cx2, cx3 = 2.0, 5.5, 9.0
    bw = 3.40          # box width

    # ================================================================ E1 line
    box(ax, cx1, yT, bw, h1, FILL["e1"], EDGE["e1"])
    txt(ax, cx1, yT + 0.16, "E1 · Passing-style &\nevent-frequency features",
        size=9.0, weight="bold", color=EDGE["e1"])
    txt(ax, cx1, yT - 0.22, "15 informative (3 pass-space + 12 event)",
        size=7.5, color="0.15")
    arrow(ax, cx1, yT - half, cx1, yM + half)

    box(ax, cx1, yM, bw, h1, FILL["e1"], EDGE["e1"])
    txt(ax, cx1, yM + 0.16, "PCA  →  k-means", size=9.0, weight="bold", color=EDGE["e1"])
    txt(ax, cx1, yM - 0.22, "11 comps · 95.9% var · k=3 · sil 0.134",
        size=7.5, color="0.15")
    arrow(ax, cx1, yM - half, cx1, yB + half)

    box(ax, cx1, yB, bw, h1, FILL["e1"], EDGE["e1"], lw=1.4)
    txt(ax, cx1, yB + 0.16, "Tactical archetypes", size=9.0, weight="bold", color=EDGE["e1"])
    txt(ax, cx1, yB - 0.22, "C0 transition · C1 turnover · C2 possession",
        size=7.5, color="0.15")

    # ================================================================ E2 line
    box(ax, cx2, yT, bw, h1, FILL["e2"], EDGE["e2"])
    txt(ax, cx2, yT + 0.16, "E2 · Freeze-frame\nspatial features",
        size=9.0, weight="bold", color=EDGE["e2"])
    txt(ax, cx2, yT - 0.22, "defensive structure · pressing · lanes",
        size=7.5, color="0.15")
    arrow(ax, cx2, yT - half, cx2, yM + half)

    box(ax, cx2, yM, bw, h1, FILL["e2"], EDGE["e2"])
    txt(ax, cx2, yM + 0.16, "Complementarity\nvs event streams", size=9.0,
        weight="bold", color=EDGE["e2"])
    txt(ax, cx2, yM - 0.22, "mean |r| ≤ 0.20 · clustering ARI ≈ 0", size=7.5,
        color="0.15")
    arrow(ax, cx2, yM - half, cx2, yB + half)

    box(ax, cx2, yB, bw, h1, FILL["e2"], EDGE["e2"], lw=1.4)
    txt(ax, cx2, yB + 0.16, "Independent spatial axis", size=9.0, weight="bold",
        color=EDGE["e2"])
    txt(ax, cx2, yB - 0.22, "orthogonal to event-stream tactics", size=7.5,
        color="0.15")

    # ================================================================ E3 line
    box(ax, cx3, yT, bw, h1, FILL["e3"], EDGE["e3"])
    txt(ax, cx3, yT + 0.16, "E3 · Passing-network\ngraph",
        size=9.0, weight="bold", color=EDGE["e3"])
    txt(ax, cx3, yT - 0.22, "20 players · 16-d node features", size=7.5, color="0.15")
    arrow(ax, cx3, yT - half, cx3, yM + half)

    box(ax, cx3, yM, bw, h1, FILL["e3"], EDGE["e3"])
    txt(ax, cx3, yM + 0.16, "→ GCN outcome model\n(Fig. 2)",
        size=9.0, weight="bold", color=EDGE["e3"])
    txt(ax, cx3, yM - 0.22, "graph representation of tactics", size=7.5, color="0.15")
    arrow(ax, cx3, yM - half, cx3, yB + half)

    box(ax, cx3, yB, bw, h1, FILL["e3"], EDGE["e3"], lw=1.4)
    txt(ax, cx3, yB + 0.16, "Outcome prediction\n(W / D / L)", size=9.0,
        weight="bold", color=EDGE["e3"])
    txt(ax, cx3, yB - 0.22, "see Results §4.3", size=7.5, color="0.15")

    # arrows from the three outputs down to the validation box
    for cx in (cx1, cx2, cx3):
        arrow(ax, cx, yB - half, cx, val_cy + 0.825)

    # ================================================================ validation
    box(ax, 5.5, val_cy, 10.4, 1.65, FILL["val"], EDGE["val"], lw=1.1)
    txt(ax, 5.5, val_cy + 0.555, "Cross-domain validation", size=9.5,
        weight="bold", color=EDGE["val"])
    txt(ax, 5.5, val_cy + 0.205, "gender transfer:   male → female  0.544",
        size=8.0, color="0.15")
    txt(ax, 5.5, val_cy - 0.145,
        "league transfer:  La Liga 0.528 · Ligue 1 0.509 · Serie A 0.495 · PL 0.491",
        size=8.0, color="0.15")
    txt(ax, 5.5, val_cy - 0.475,
        "→ transferable tactical structure: no consistent degradation",
        size=8.0, style="italic", weight="bold", color=OI["grey"])

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig_overview.pdf")
    fig.savefig(f"{out_dir}/fig_overview.png", dpi=300)
    plt.close(fig)
    print(f"[saved] {out_dir}/fig_overview.pdf / .png")


if __name__ == "__main__":
    draw_overview()
