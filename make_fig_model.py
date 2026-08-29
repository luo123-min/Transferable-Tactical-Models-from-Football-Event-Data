"""
make_fig_model.py — Model architecture diagram for E3 (trained GCN).

Horizontal (landscape) two-row layout; main data flow left → right.

  Upper row:  X (top)                         GCN layer 1 → GCN layer 2 →  mean
              A (bottom) → Â (sym. norm.) ──→                       max   → Graph emb
                                                                        sum
  Lower row:  Style (22-d) → Concat (406-d) → Linear+ReLU → Linear+Softmax → Outcome
                  ↑                          ↑
                  event-stream (E1/E2)     vertical from Graph emb

Aesthetic v7 (journal figure*, ~15.7×6.1 in):
  - "card" style: soft drop shadow + white fill + colored 1.2 border; rounded
    corners; colored bold title + caption lines. No in-figure title or training
    bar — those live in the figure caption + Methods text.
  - two faint background bands ("GRAPH → REPRESENTATION",
    "FUSION → PREDICTION") with small label tabs, framing the pipeline.
  - curved FancyArrowPatch connectors (rounded caps) instead of straight lines.
  - the residual arc is explained in the figure caption, not drawn in-figure.

Style matches make_figures.py / make_fig_overview.py (Okabe-Ito palette,
DejaVu Sans, pdf.fonttype=42, 300 dpi). Outputs:
  figures_paper/fig_model.pdf (vector) + fig_model.png (300 dpi).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "figures_paper"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9.0,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "mathtext.fontset": "dejavusans",
})
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
      "verm": "#D55E00", "sky": "#56B4E9", "purple": "#CC79A7",
      "yellow": "#F0E442", "black": "#000000", "grey": "#666666"}


def card(ax, cx, cy, w, h, ec, title, lines, accent=None, radius=0.10,
         shadow=True):
    """A modern rounded card: shadow + white fill + colored border + accent."""
    acc = accent or ec
    x0, y0 = cx - w / 2, cy - h / 2
    if shadow:
        ax.add_patch(FancyBboxPatch((x0 + 0.06, y0 - 0.06), w, h,
                       boxstyle=f"round,pad=0.02,rounding_size={radius}",
                       linewidth=0, facecolor="#9AA0A6", alpha=0.18,
                       zorder=1))
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                   boxstyle=f"round,pad=0.02,rounding_size={radius}",
                   linewidth=1.2, edgecolor=ec, facecolor="white",
                   zorder=2))
    # title
    ax.text(cx, cy + h / 2 - 0.15, title, fontsize=8.5, color=ec,
            fontweight="bold", ha="center", va="center", zorder=4)
    # body lines  (each entry: (text, fontsize, color))
    n = len(lines)
    if n == 0:
        return
    top = y0 + h - 0.30
    bot = y0 + 0.10
    if n == 1:
        ys = [(top + bot) / 2]
    else:
        ys = [top - i * (top - bot) / (n - 1) for i in range(n)]
    for i, (s, fs, col) in enumerate(lines):
        ax.text(cx, ys[i], s, fontsize=fs, color=col, ha="center",
                va="center", zorder=4)


def bar(ax, cx, cy, w, h, ec, text, radius=0.08):
    """A slim bordered bar (no accent) for the training configuration."""
    x0, y0 = cx - w / 2, cy - h / 2
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                   boxstyle=f"round,pad=0.02,rounding_size={radius}",
                   linewidth=1.0, edgecolor=ec, facecolor="#FAFBFC",
                   zorder=2))
    ax.text(cx, cy, text, fontsize=7.5, color="0.25", ha="center",
            va="center", zorder=4)


def band(ax, x0, y0, w, h, fill, label, lab_color, radius=0.18):
    """Faint background panel grouping a pipeline stage."""
    ax.add_patch(FancyBboxPatch((x0, y0), w, h,
                   boxstyle=f"round,pad=0.02,rounding_size={radius}",
                   linewidth=0, facecolor=fill, alpha=1.0, zorder=0))
    tw, th = 2.85, 0.30
    tx, ty = x0 + 0.18, y0 + h - 0.18
    ax.add_patch(FancyBboxPatch((tx, ty - th / 2), tw, th,
                   boxstyle="round,pad=0.02,rounding_size=0.05",
                   linewidth=0, facecolor=lab_color, alpha=0.92, zorder=1))
    ax.text(tx + tw / 2, ty, label, fontsize=7.5, color="white",
            fontweight="bold", ha="center", va="center", zorder=2)


def carrow(ax, p1, p2, color="0.4", lw=1.0, rad=0.0, ls="-"):
    """Curved arrow connector with rounded cap."""
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                   connectionstyle=f"arc3,rad={rad}", linewidth=lw,
                   edgecolor=color, facecolor=color, linestyle=ls,
                   shrinkA=1.5, shrinkB=2.5, zorder=5))


def draw_model(out_dir=OUT):
    """Draw the E3 model architecture and save PDF+PNG."""
    fig, ax = plt.subplots(figsize=(15.7, 5.0))
    ax.set_xlim(0, 15.7)
    ax.set_ylim(0.55, 5.55)
    ax.axis("off")

    # ---- background bands ------------------------------------------------
    band(ax, 0.15, 2.95, 15.40, 2.40, "#EEF2F5",
         "GRAPH  \u2192  REPRESENTATION", OI["blue"])
    band(ax, 0.15, 0.75, 15.45, 1.15, "#F7F1F7",
         "FUSION  \u2192  PREDICTION", OI["purple"])

    # ---- vertical positions (no in-figure title / training bar:
    #      these live in the figure caption + Methods text) ---------------
    y_hi   = 4.55
    y_mid  = 4.00
    y_lo   = 3.55
    y_low  = 1.40

    # ---- horizontal positions (left → right) ----------------------------
    x_in   = 0.95
    x_norm = 2.50
    x_g1   = 4.425
    x_g2   = 6.625
    x_ro   = 8.45
    x_emb  = 10.10
    x_sty  = 8.15
    x_cat  = 10.10
    x_h1   = 11.80
    x_h2   = 13.40
    x_out  = 14.85

    # ---- box dimensions --------------------------------------------------
    w_in,   h_in   = 1.10, 0.54
    w_norm, h_norm = 1.20, 0.44
    w_g,    h_g    = 1.75, 0.72
    w_ro,   h_ro   = 1.00, 0.44
    w_emb,  h_emb  = 1.40, 0.50
    w_sty,  h_sty  = 1.40, 0.60
    w_cat,  h_cat  = 1.40, 0.54
    w_h,    h_h    = 1.20, 0.50
    w_out,  h_out  = 0.90, 0.46

    # ================================================================ inputs
    card(ax, x_in, y_hi, w_in, h_in, OI["black"], "Node features $X$",
         [("20 $\\times$ 16", 8.0, "0.15"), ("per team-match", 7.5, "0.4")],
         accent=OI["grey"])
    card(ax, x_in, y_lo, w_in, h_in, OI["black"], "Adjacency $A$",
         [("20 $\\times$ 20", 8.0, "0.15"), ("binary + self-loops", 7.5, "0.4")],
         accent=OI["grey"])

    # ================================================================ normalization
    card(ax, x_norm, y_lo, w_norm, h_norm, OI["blue"],
         r"$\hat{A}=D^{-1/2}AD^{-1/2}$", [("sym. norm.", 7.5, "0.3")],
         accent=OI["blue"], radius=0.08)

    # ================================================================ GCN layer 1
    card(ax, x_g1, y_mid, w_g, h_g, OI["blue"], "GCN layer 1",
         [("$H^{(1)}{=}\\mathrm{ReLU}(\\mathrm{LN}(\\hat{A}XW^{(1)})))$", 7.5, "0.15"),
          ("16 $\\to$ 128", 8.0, "0.2")], accent=OI["blue"])

    # ================================================================ GCN layer 2
    card(ax, x_g2, y_mid, w_g, h_g, OI["blue"], "GCN layer 2",
         [("$H^{(2)}{=}\\mathrm{ReLU}(\\mathrm{LN}(\\hat{A}H^{(1)}W^{(2)})))$", 7.5, "0.15"),
          ("128 $\\to$ 128", 8.0, "0.2")], accent=OI["blue"])

    # ================================================================ residual arc
    g_top = y_mid + h_g / 2
    ar_h  = g_top + 0.42
    ax.plot([x_g1, x_g1, x_g2, x_g2], [g_top, ar_h, ar_h, g_top],
            color=OI["orange"], lw=1.2, solid_capstyle="round", zorder=1)
    ax.add_patch(FancyArrowPatch((x_g2, ar_h), (x_g2, g_top),
                   arrowstyle="-|>", mutation_scale=11, linewidth=1.2,
                   edgecolor=OI["orange"], facecolor=OI["orange"],
                   shrinkA=0, shrinkB=1.5, zorder=2))
    ax.add_patch(FancyBboxPatch(((x_g1 + x_g2) / 2 - 0.55, ar_h + 0.04),
                   1.10, 0.26, boxstyle="round,pad=0.02,rounding_size=0.12",
                   linewidth=0, facecolor="white", alpha=0.92, zorder=3))
    ax.text((x_g1 + x_g2) / 2, ar_h + 0.17, "+ residual", size=8.0,
            color=OI["orange"], style="italic", ha="center", va="center",
            zorder=4)

    # ================================================================ readout (3)
    # dedicated even vertical spacing (decoupled from the input column) so the
    # three pooling boxes keep a clear, consistent gap
    y_ro = [4.64, 4.00, 3.36]
    ro_labels = ["mean", "max", "sum"]
    for yr, lbl in zip(y_ro, ro_labels):
        card(ax, x_ro, yr, w_ro, h_ro, OI["green"], lbl,
             [("128-d", 7.5, "0.2")], accent=OI["green"], radius=0.08)

    # ================================================================ graph embedding
    card(ax, x_emb, y_mid, w_emb, h_emb, OI["green"], "Graph embedding",
         [("384-d", 8.0, "0.2")], accent=OI["green"])

    # ================================================================ style descriptor
    card(ax, x_sty, y_low, w_sty, h_sty, OI["orange"], "Style descriptor",
         [("22-d", 8.0, OI["orange"]),
          ("$\\leftarrow$ event-stream (E1/E2)", 7.0, "0.3")],
         accent=OI["orange"])

    # ================================================================ concatenate
    card(ax, x_cat, y_low, w_cat, h_cat, OI["orange"], "Concatenate",
         [("384 + 22 $\\to$ 406", 7.5, "0.2")], accent=OI["orange"])

    # ================================================================ head MLP
    card(ax, x_h1, y_low, w_h, h_h, OI["purple"], "Linear + ReLU",
         [("406 $\\to$ 64", 8.0, "0.2"), ("dropout", 7.5, "0.3")],
         accent=OI["purple"])
    card(ax, x_h2, y_low, w_h, h_h, OI["purple"], "Linear + Softmax",
         [("64 $\\to$ 3", 8.0, "0.2")], accent=OI["purple"])

    # ================================================================ outcome
    card(ax, x_out, y_low, w_out, h_out, OI["verm"], "Outcome",
         [("W   D   L", 8.5, "0.15")], accent=OI["verm"], radius=0.08)

    # ================================================================ arrows
    carrow(ax, (x_in + w_in / 2, y_hi), (x_g1 - w_g / 2, y_mid + 0.12), rad=0.10)
    carrow(ax, (x_in + w_in / 2, y_lo), (x_norm - w_norm / 2, y_lo))
    carrow(ax, (x_norm + w_norm / 2, y_lo), (x_g1 - w_g / 2, y_mid - 0.12),
           color=OI["blue"], rad=0.10)
    carrow(ax, (x_g1 + w_g / 2, y_mid), (x_g2 - w_g / 2, y_mid))
    for yr in y_ro:
        carrow(ax, (x_g2 + w_g / 2, y_mid), (x_ro - w_ro / 2, yr), rad=0.12)
    for yr in y_ro:
        carrow(ax, (x_ro + w_ro / 2, yr), (x_emb - w_emb / 2, y_mid),
               color=OI["green"], rad=0.12)
    carrow(ax, (x_emb, y_mid - h_emb / 2), (x_cat, y_low + h_cat / 2), rad=0.05)
    carrow(ax, (x_sty + w_sty / 2, y_low), (x_cat - w_cat / 2, y_low),
           color=OI["orange"], ls="--")
    carrow(ax, (x_cat + w_cat / 2, y_low), (x_h1 - w_h / 2, y_low))
    carrow(ax, (x_h1 + w_h / 2, y_low), (x_h2 - w_h / 2, y_low))
    carrow(ax, (x_h2 + w_h / 2, y_low), (x_out - w_out / 2, y_low))

    fig.tight_layout()
    for name in ("fig_model.pdf", "fig_model.png"):
        fig.savefig(os.path.join(out_dir, name), dpi=300)
    plt.close(fig)
    print(f"Saved {os.path.join(out_dir, 'fig_model.pdf')} and .png")


if __name__ == "__main__":
    draw_model()
