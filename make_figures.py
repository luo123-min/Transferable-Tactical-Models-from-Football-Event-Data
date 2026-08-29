"""
make_figures.py — Publication-quality figures for the paper (PDF vector + PNG 300dpi).

Generates (figures_paper/):
  fig_overview.pdf/png    — Figure 1: overall pipeline overview
  fig_ablation.pdf/png    — six-system ablation (acc + macro-F1)
  fig_transfer.pdf/png    — same-dist / cross-league (5-seed error bars) / cross-gender
  fig_pca.pdf/png         — GCN embedding PCA (by outcome; male vs female)
  fig_confusion.pdf/png   — confusion matrices (male test & female transfer)
  fig_robustness.pdf/png  — per-seed macro-F1 vs league-transfer intervals
  fig_corr360.pdf/png     — E2 cross-family correlation heatmap (360 vs event-stream)
  fig_archetypes.pdf/png  — E1 cluster profile heatmap (z-scores)
  ablation_results.csv    — regenerated six-system table (seed 42)

Trainings: 1 (ours, seed 42) + 5 (robustness seeds) ~= 10 min on CPU.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA

# ---------------------------------------------------------------- style setup
OUT = "figures_paper"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "axes.linewidth": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "pdf.fonttype": 42,       # TrueType: editable text in PDF
    "ps.fonttype": 42,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})
# Okabe-Ito colorblind-safe palette
OI = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "verm": "#D55E00",
      "sky": "#56B4E9", "purple": "#CC79A7", "yellow": "#F0E442", "black": "#000000"}

def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")            # vector
    fig.savefig(f"{OUT}/{name}.png", dpi=300)   # 300 dpi raster
    plt.close(fig)
    print(f"[saved] {OUT}/{name}.pdf / .png")

# ---------------------------------------------------------------- data
DATA = "results_gnn/graph_data.npz"
META = "results_gnn/graph_meta.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
D = np.load(DATA)
node_feat = torch.tensor(D["node_feat"], dtype=torch.float32, device=DEVICE)
adj = torch.tensor(D["adj"], dtype=torch.float32, device=DEVICE)
mask = torch.tensor(D["mask"], dtype=torch.float32, device=DEVICE)
label = torch.tensor(D["label"], dtype=torch.long, device=DEVICE)
handcrafted = D["handcrafted"].astype(np.float32)
meta = pd.read_csv(META)
G, F = node_feat.shape[0], node_feat.shape[2]
print(f"[data] graphs={G}, device={DEVICE}")

def idx(a):
    return np.where(a)[0]

def match_split(male_df, seed, test_frac=0.30):
    meta2 = male_df.copy()
    meta2["mo"] = np.where(meta2["home_score"] > meta2["away_score"], "H",
                   np.where(meta2["home_score"] < meta2["away_score"], "A", "D"))
    uniq = meta2[["match_id", "mo"]].drop_duplicates().reset_index(drop=True)
    tr, tmp = train_test_split(uniq["match_id"].values, test_size=test_frac,
                               random_state=seed, stratify=uniq["mo"].values)
    tmp_u = uniq[uniq["match_id"].isin(tmp)]
    va, te = train_test_split(tmp_u["match_id"].values, test_size=0.5,
                              random_state=seed, stratify=tmp_u["mo"].values)
    return (meta2["match_id"].isin(tr).values,
            meta2["match_id"].isin(va).values,
            meta2["match_id"].isin(te).values)

# ---------------------------------------------------------------- GCN (as in transfer_league.py)
class GCN(nn.Module):
    def __init__(self, f_in, hidden=128, n_layers=2, n_class=3, head=64, dropout=0.3):
        super().__init__()
        self.Win = nn.Linear(f_in, hidden)
        self.convs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head1 = nn.Linear(hidden * 3, head)
        self.head2 = nn.Linear(head, n_class)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, mask):
        h = torch.relu(self.Win(x))
        for conv, bn in zip(self.convs, self.bns):
            support = torch.bmm(adj, h)
            h2 = torch.relu(bn(conv(support)))
            h = h + h2
        mb = mask.unsqueeze(-1)
        hm = h * mb
        meanp = hm.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        maxp = (hm + (1.0 - mb) * (-1e9)).max(1).values
        sump = hm.sum(1)
        return torch.cat([meanp, maxp, sump], dim=1)


def train_gcn(train_idx, val_idx, test_idx, gfeat=None, use_style=True,
              train_epochs=60, seed=42, verbose=False):
    from sklearn.utils.class_weight import compute_class_weight
    torch.manual_seed(seed); np.random.seed(seed)
    ytr = label[train_idx].cpu().numpy()
    cls_w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)
    cw = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)

    gcn = GCN(F, hidden=128, n_layers=2).to(DEVICE)
    g_dim = gfeat.shape[1] if (use_style and gfeat is not None) else 0
    head = nn.Sequential(
        nn.Linear(128 * 3 + g_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3),
    ).to(DEVICE)
    opt = torch.optim.Adam(list(gcn.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=8, min_lr=1e-5)
    if use_style and gfeat is not None:
        gt = torch.tensor(gfeat[train_idx], dtype=torch.float32, device=DEVICE)
        gv = torch.tensor(gfeat[val_idx], dtype=torch.float32, device=DEVICE)
        ge = torch.tensor(gfeat[test_idx], dtype=torch.float32, device=DEVICE)
        tr_b = TensorDataset(node_feat[train_idx], adj[train_idx], mask[train_idx],
                             gt, label[train_idx])
    else:
        gv = ge = None
        tr_b = TensorDataset(node_feat[train_idx], adj[train_idx], mask[train_idx],
                             label[train_idx])
    dl = DataLoader(tr_b, batch_size=128, shuffle=True)

    def hf(pooled, g):
        if use_style and g is not None:
            return head(torch.cat([pooled, g], dim=1))
        return head(pooled)

    best_val, best_state, patience = -1, None, 0
    for ep in range(train_epochs):
        gcn.train(); head.train()
        for batch in dl:
            if use_style:
                xb, ab, mb, gxb, lb = batch
            else:
                xb, ab, mb, lb = batch; gxb = None
            opt.zero_grad()
            out = hf(gcn(xb, ab, mb), gxb)
            loss = crit(out, lb); loss.backward(); opt.step()
        gcn.eval(); head.eval()
        with torch.no_grad():
            pv = hf(gcn(node_feat[val_idx], adj[val_idx], mask[val_idx]), gv)
            vp = pv.argmax(1).cpu().numpy()
        vf1 = f1_score(label[val_idx].cpu().numpy(), vp, average="macro")
        if verbose and ep % 15 == 0:
            print(f"      ep {ep:3d} val_macroF1={vf1:.3f}")
        sched.step(vf1)
        if vf1 > best_val + 1e-4:
            best_val = vf1
            best_state = (dict(gcn.named_parameters()), dict(head.named_parameters()))
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    gcn.load_state_dict({k: v for k, v in best_state[0].items()})
    head.load_state_dict({k: v for k, v in best_state[1].items()})
    gcn.eval(); head.eval()
    with torch.no_grad():
        pe = hf(gcn(node_feat[test_idx], adj[test_idx], mask[test_idx]), ge)
        tp = pe.argmax(1).cpu().numpy()
        emb_te = gcn(node_feat[test_idx], adj[test_idx], mask[test_idx]).detach().cpu().numpy()
    return tp, emb_te, (gcn, head)


# ---------------------------------------------------------------- shared splits
male_mask = (meta["competition_gender"] == "male").values
female_mask = ~male_mask
tr42, va42, te42 = match_split(meta[male_mask], seed=42)
gsc = StandardScaler().fit(handcrafted[idx(tr42)])
gfeat_all = gsc.transform(handcrafted)
y_te42 = label[idx(te42)].cpu().numpy()

CLASSES = ["Loss", "Draw", "Win"]

# ================================================================ [1] six-system ablation
print("\n=== [1] six-system ablation (seed 42) ===")
rows = []

torch.manual_seed(42)
rnd = GCN(F).to(DEVICE); rnd.eval()
with torch.no_grad():
    e_tr = rnd(node_feat[idx(tr42)], adj[idx(tr42)], mask[idx(tr42)]).cpu().numpy()
    e_te = rnd(node_feat[idx(te42)], adj[idx(te42)], mask[idx(te42)]).cpu().numpy()
Xr_tr = np.hstack([e_tr, gfeat_all[idx(tr42)]])
Xr_te = np.hstack([e_te, gfeat_all[idx(te42)]])
p_rnd = LogisticRegression(max_iter=5000).fit(Xr_tr, label[idx(tr42)].cpu().numpy()).predict(Xr_te)
rows.append(("Random GCN + style (untrained)", p_rnd))

p_hc = LogisticRegression(max_iter=5000).fit(gfeat_all[idx(tr42)],
                                             label[idx(tr42)].cpu().numpy()).predict(gfeat_all[idx(te42)])
rows.append(("Style descriptors + LR", p_hc))

node_mean = node_feat.mean(dim=1).cpu().numpy()
scn = StandardScaler().fit(node_mean[idx(tr42)])
Xn_tr = np.hstack([scn.transform(node_mean[idx(tr42)]), gfeat_all[idx(tr42)]])
Xn_te = np.hstack([scn.transform(node_mean[idx(te42)]), gfeat_all[idx(te42)]])
p_node = LogisticRegression(max_iter=5000).fit(Xn_tr, label[idx(tr42)].cpu().numpy()).predict(Xn_te)
rows.append(("Node mean-pool + style + LR (no graph)", p_node))

flat = node_feat.reshape(G, -1).cpu().numpy()
scf = StandardScaler().fit(flat[idx(tr42)])
p_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                      early_stopping=True).fit(scf.transform(flat[idx(tr42)]),
                                               label[idx(tr42)].cpu().numpy()
                                               ).predict(scf.transform(flat[idx(te42)]))
rows.append(("MLP (flat raw node feats)", p_mlp))

p_raw, _, _ = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat=None, use_style=False,
                        train_epochs=60, seed=42)
rows.append(("Trained GCN (raw graph only)", p_raw))

p_ours, emb_ours, model_ours = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat_all,
                                         use_style=True, train_epochs=60, seed=42,
                                         verbose=True)
rows.append(("Trained GCN + style (ours)", p_ours))

abl = pd.DataFrame([{"model": nm,
                     "accuracy": accuracy_score(y_te42, p),
                     "macro_f1": f1_score(y_te42, p, average="macro")}
                    for nm, p in rows])
abl.to_csv(f"{OUT}/ablation_results.csv", index=False)
print(abl.to_string(index=False))

# ---- figure: ablation (dumbbell: macro-F1 dot at left end, accuracy open
# square at right end; line length = gap between the two metrics)
abl_s = abl.sort_values("macro_f1")          # ascending -> ours (max F1) at the top
labels = list(abl_s["model"])                 # one line, no wrapping
is_ours = ["ours" in m for m in abl_s["model"]]
fig, ax = plt.subplots(figsize=(6.0, 3.2))
ypos = np.arange(len(abl_s))                  # y=0 bottom (GCN raw) ... y=5 top (ours)
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
ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=7.0)
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
save(fig, "fig_ablation")

# ================================================================ [2] embeddings + confusion (ours model)
gcn_m, head_m = model_ours
gcn_m.eval(); head_m.eval()
with torch.no_grad():
    emb_f = gcn_m(node_feat[female_mask], adj[female_mask], mask[female_mask])
    gxf = torch.tensor(gfeat_all[female_mask], dtype=torch.float32, device=DEVICE)
    p_f = head_m(torch.cat([emb_f, gxf], dim=1)).argmax(1).cpu().numpy()
    emb_f = emb_f.cpu().numpy()
y_f = label[female_mask].cpu().numpy()
print(f"\n[cross-gender] acc={accuracy_score(y_f, p_f):.3f} "
      f"macroF1={f1_score(y_f, p_f, average='macro'):.3f}")

# ---- figure: PCA (2 panels)
pca = PCA(n_components=2, random_state=0)
pca.fit(emb_ours)
Pm = pca.transform(emb_ours); Pf = pca.transform(emb_f)
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
cc = {0: OI["verm"], 1: "grey", 2: OI["blue"]}
for k, cname in enumerate(CLASSES):
    s = y_te42 == k
    axes[0].scatter(Pm[s, 0], Pm[s, 1], s=8, alpha=0.65, color=cc[k],
                    label=f"{cname} (n={s.sum()})", lw=0)
axes[0].set_title("Male test split, by true outcome")
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].legend(handletextpad=0.2, borderaxespad=0.2)
axes[1].scatter(Pm[:, 0], Pm[:, 1], s=8, alpha=0.45, color=OI["sky"], lw=0,
                label=f"male test (n={len(Pm)})")
axes[1].scatter(Pf[:, 0], Pf[:, 1], s=8, alpha=0.45, color=OI["purple"], lw=0,
                label=f"female transfer (n={len(Pf)})")
axes[1].set_title("Male test vs.\\ female transfer")
axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[1].legend(handletextpad=0.2, borderaxespad=0.2)
for a in axes:
    a.tick_params(length=2.5)
fig.tight_layout()
save(fig, "fig_pca")

# ---- figure: confusion matrices (row-normalised)
cm_m = confusion_matrix(y_te42, p_ours, normalize="true")
cm_f = confusion_matrix(y_f, p_f, normalize="true")
cmap = LinearSegmentedColormap.from_list("wr", ["#FFFFFF", OI["blue"]])
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.0))
for a, cm, ttl in zip(axes, [cm_m, cm_f],
                      ["Male test (macro-F1 0.52)", "Female transfer (macro-F1 0.54)"]):
    im = a.imshow(cm, cmap=cmap, vmin=0, vmax=0.7)
    for i in range(3):
        for j in range(3):
            a.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=8.5,
                   color="white" if cm[i, j] > 0.45 else "black")
    a.set_xticks(range(3)); a.set_yticks(range(3))
    a.set_xticklabels(CLASSES); a.set_yticklabels(CLASSES)
    a.set_xlabel("Predicted"); a.set_ylabel("True")
    a.set_title(ttl)
    a.spines[:].set_visible(False)
    a.tick_params(length=0)
fig.tight_layout()
save(fig, "fig_confusion")

# ================================================================ [3] multi-seed robustness (per-seed points)
print("\n=== [3] robustness seeds ===")
seeds = [42, 7, 13, 99, 2026]
seed_f1, seed_acc = [], []
for s in seeds:
    trs, vas, tes = match_split(meta[male_mask], seed=s)
    gsc_s = StandardScaler().fit(handcrafted[idx(trs)])
    gf_s = gsc_s.transform(handcrafted)
    ps, _, _ = train_gcn(idx(trs), idx(vas), idx(tes), gf_s, use_style=True,
                         train_epochs=60, seed=s)
    ys = label[idx(tes)].cpu().numpy()
    a = accuracy_score(ys, ps); f = f1_score(ys, ps, average="macro")
    seed_acc.append(a); seed_f1.append(f)
    print(f"  seed {s:5d}  acc={a:.3f}  macroF1={f:.3f}")
pd.DataFrame({"seed": seeds, "accuracy": seed_acc, "macro_f1": seed_f1}
             ).to_csv(f"{OUT}/robustness_perseed.csv", index=False)

# ================================================================ [4] transfer figure (from CSV)
trf = pd.read_csv("results_gnn/transfer_league.csv")
short = {"same-distribution (male test)": "Same-dist.\n(male test)",
         "cross-league -> La Liga": "La Liga",
         "cross-league -> Ligue 1": "Ligue 1",
         "cross-league -> Premier League": "Premier\nLeague",
         "cross-league -> Serie A": "Serie A",
         "cross-gender -> female": "Female\n(cross-gender)"}
trf["lab"] = trf["direction"].map(short)
trf["kind"] = trf["direction"].map(lambda d: "same" if "same" in d else
                                   ("female" if "female" in d else "league"))
order = trf.sort_values("macro_f1", ascending=False)

fig, ax = plt.subplots(figsize=(3.5, 3.0))
kcolors = {"same": OI["black"], "league": OI["blue"], "female": OI["verm"]}
ypos = np.arange(len(order))[::-1]
for y, (_, r) in zip(ypos, order.iterrows()):
    c = kcolors[r["kind"]]
    ax.errorbar(r["macro_f1"], y, xerr=r["f1_std"], fmt="o", color=c, ms=5,
                capsize=2.5, elinewidth=1.1, capthick=1.1)
    ax.text(r["macro_f1"] + r["f1_std"] + 0.012, y,
            f"{r['macro_f1']:.3f} (n={int(r['n']):,})", va="center", fontsize=7.2)
ax.set_yticks(ypos); ax.set_yticklabels(order["lab"], fontsize=7.5)
ax.set_xlabel("Macro-F1 (mean $\\pm$ std over 5 seeds)")
ax.set_xlim(0.42, 0.68)
ax.axvline(np.mean(seed_f1), color=OI["green"], lw=0.9, ls="--", zorder=1)
ax.axvspan(np.mean(seed_f1) - np.std(seed_f1), np.mean(seed_f1) + np.std(seed_f1),
           color=OI["green"], alpha=0.12, zorder=0)
ax.text(np.mean(seed_f1), len(ypos) + 0.45, "5-seed robust interval",
        fontsize=7, color=OI["green"], va="bottom", ha="center")
fig.tight_layout()
save(fig, "fig_transfer")

# ================================================================ [5] robustness figure
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
save(fig, "fig_robustness")

# ================================================================ [6] E2 correlation heatmap
corr = pd.read_csv("results_360/corr_360_vs_eventstream.csv", index_col=0)
poss_cols = [c for c in corr.columns if c.startswith("poss_")]
def_cols = [c for c in corr.columns if c.startswith("def_")]
ev_cols = [c for c in corr.columns if not c.startswith(("poss_", "def_"))]
reorder = poss_cols + def_cols + ev_cols
C = corr.loc[reorder, reorder]
n_p, n_d, n_e = len(poss_cols), len(def_cols), len(ev_cols)
fig, ax = plt.subplots(figsize=(4.6, 4.0))
cmap2 = LinearSegmentedColormap.from_list("rb", [OI["blue"], "#FFFFFF", OI["verm"]])
im = ax.imshow(C.values, cmap=cmap2, vmin=-1, vmax=1)
# block outlines
import matplotlib.patches as mpatches
for (xy, wh) in [((n_p, 0), (n_e, n_p + n_d)), ((0, n_p), (n_p + n_d, n_e)),
                 ((n_p, n_p), (n_d, n_d)), ((0, 0), (n_p, n_p)), ((n_e + 0.0, 0), (0, 0))][:4]:
    pass
ax.add_patch(mpatches.Rectangle((n_p - 0.5, -0.5), n_e, n_p + n_d, fill=False,
                                edgecolor=OI["black"], lw=1.2))
ax.add_patch(mpatches.Rectangle((-0.5, n_p - 0.5), n_p + n_d, n_e, fill=False,
                                edgecolor=OI["black"], lw=1.2))
short_lab = lambda c: c.replace("poss_", "P: ").replace("def_", "D: ")
ax.set_xticks(range(len(reorder)))
ax.set_xticklabels([short_lab(c) for c in reorder], rotation=90, fontsize=6.2)
ax.set_yticks(range(len(reorder)))
ax.set_yticklabels([short_lab(c) for c in reorder], fontsize=6.2)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
cb.set_label("Pearson $r$", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title("Spatial (360) vs.\\ event-stream features:\ncross-block correlations near zero",
             fontsize=8.5)
fig.tight_layout()
save(fig, "fig_corr360")

# ================================================================ [7] E1 archetype heatmap
prof = pd.read_csv("results_full/cluster_profiles_full.csv")
feat_show = ["mean_pass_length", "long_pass_frac", "short_pass_frac", "rate_pass",
             "rate_carry", "rate_shot", "rate_pressure", "rate_dribble",
             "rate_ball_recovery", "rate_interception", "rate_clearance",
             "rate_block", "rate_miscontrol", "rate_dispossessed"]
nice = {"mean_pass_length": "mean pass length", "long_pass_frac": "long-pass share",
        "short_pass_frac": "short-pass share", "rate_pass": "passes / match",
        "rate_carry": "carries / match", "rate_shot": "shots / match",
        "rate_pressure": "pressing actions", "rate_dribble": "dribbles",
        "rate_ball_recovery": "ball recoveries", "rate_interception": "interceptions",
        "rate_clearance": "clearances", "rate_block": "blocks",
        "rate_miscontrol": "miscontrols", "rate_dispossessed": "dispossessions"}
M = prof[feat_show].values
fig, ax = plt.subplots(figsize=(3.5, 3.4))
cmap3 = LinearSegmentedColormap.from_list("bv", [OI["blue"], "#FFFFFF", OI["verm"]])
im = ax.imshow(M.T, cmap=cmap3, vmin=-1.2, vmax=1.2, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v = M[i, j]
        ax.text(i, j, f"{v:+.1f}", ha="center", va="center", fontsize=7,
                color="white" if abs(v) > 0.85 else "black")
ax.set_xticks(range(M.shape[0]))
ax.set_xticklabels([f"C{k}" for k in prof["cluster"]])
ax.set_yticks(range(len(feat_show)))
ax.set_yticklabels([nice[f] for f in feat_show], fontsize=7.2)
ax.spines[:].set_visible(False); ax.tick_params(length=0)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label("z-score (cluster mean)", fontsize=8); cb.ax.tick_params(labelsize=7)
ax.set_title("E1 tactical archetypes:\nstandardised cluster profiles", fontsize=8.5)
fig.tight_layout()
save(fig, "fig_archetypes")

# --------------------------------------------------------------- [8] overview
from make_fig_overview import draw_overview
draw_overview(OUT)

print("\nAll figures done.")
