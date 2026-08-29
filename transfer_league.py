"""
transfer_league.py — Cross-league transfer + robustness for the trained GCN.

Extends §13 with:
  1. Leave-one-league-out transfer within MALE football (La Liga / Ligue 1 /
     Premier League / Serie A), to disentangle "gender" from "club vs
     international / league structure" in the cross-gender transfer result.
  2. Multi-seed robustness: 5 seeds of the main model (Trained GCN + style)
     with fresh splits, reported as mean +/- std.
  3. McNemar significance tests: ours vs each baseline on the seed-42 test.
  4. Logistic-regression baselines rerun with max_iter=5000 (converged).
  5. FAIR no-graph baseline: per-graph mean-pooled node features + style -> LR,
     isolating the graph message-passing contribution (same node+style info,
     no convolution). Cross-league transfer now runs 5 seeds per league.

Outputs (results_gnn/):
  transfer_league.csv   — transfer matrix (same-dist / cross-league / cross-gender)
  robustness.csv        — 5-seed mean +/- std of macro-F1 / acc (ours)
  mcnemar.csv           — McNemar p-values ours vs baselines
  fig_transfer_compare.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import binom

DATA = "results_gnn/graph_data.npz"
META = "results_gnn/graph_meta.csv"
OUT = "results_gnn"
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] device={DEVICE}  torch={torch.__version__}")

D = np.load(DATA)
node_feat = torch.tensor(D["node_feat"], dtype=torch.float32, device=DEVICE)
adj = torch.tensor(D["adj"], dtype=torch.float32, device=DEVICE)
mask = torch.tensor(D["mask"], dtype=torch.float32, device=DEVICE)
label = torch.tensor(D["label"], dtype=torch.long, device=DEVICE)
handcrafted = D["handcrafted"].astype(np.float32)
meta = pd.read_csv(META)
G, F = node_feat.shape[0], node_feat.shape[2]
print(f"[data] graphs={G}  label dist={dict(zip(*np.unique(D['label'], return_counts=True)))}")


def match_split(male_df, seed, test_frac=0.30):
    """Split male matches into train / val / test by match_id (stratified).

    Matches the main protocol in train_gnn.py: 70/15/15.
    test_frac is the size of the FIRST holdout (tmp), then tmp is split
    half into val and test -> train 0.70 / val 0.15 / test 0.15.
    """
    meta2 = male_df.copy()
    meta2["mo"] = np.where(meta2["home_score"] > meta2["away_score"], "H",
                   np.where(meta2["home_score"] < meta2["away_score"], "A", "D"))
    uniq = meta2[["match_id", "mo"]].drop_duplicates().reset_index(drop=True)
    mids, mout = uniq["match_id"].values, uniq["mo"].values
    tr, tmp = train_test_split(mids, test_size=test_frac, random_state=seed, stratify=mout)
    tmp_u = uniq[uniq["match_id"].isin(tmp)]
    va, te = train_test_split(tmp_u["match_id"].values, test_size=0.5,
                              random_state=seed, stratify=tmp_u["mo"].values)
    m_tr = meta2["match_id"].isin(tr).values
    m_va = meta2["match_id"].isin(va).values
    m_te = meta2["match_id"].isin(te).values
    return m_tr, m_va, m_te


def idx(a):
    return np.where(a)[0]


# ----------------------------------------------------------------------------
# GCN (same architecture as train_gnn.py)
# ----------------------------------------------------------------------------
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
    """Train GCN (+optional style head) on train, early-stop on val, score test.
    gfeat must already be standardised on THIS fold's train split."""
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
        if verbose and (ep % 15 == 0 or ep == train_epochs - 1):
            print(f"      ep {ep:3d} loss={loss.item():.3f} val_macroF1={vf1:.3f}")
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


def mcnemar_p(y, pA, pB):
    """Exact McNemar (two-sided binomial) on paired predictions."""
    b = int(np.sum((pA == y) & (pB != y)))   # A right, B wrong
    c = int(np.sum((pA != y) & (pB == y)))   # A wrong, B right
    n = b + c
    if n == 0:
        return 1.0
    p = binom.cdf(min(b, c), n, 0.5) * 2.0
    return min(1.0, p)


# ----------------------------------------------------------------------------
# Setup: male base + seed-42 reference splits
# ----------------------------------------------------------------------------
male_mask = (meta["competition_gender"] == "male").values
female_mask = ~male_mask
tr42, va42, te42 = match_split(meta[male_mask], seed=42)

# ----------------------------------------------------------------------------
# 1. Reference 5-system ablation (seed 42) with CONVERGED LR (max_iter=5000)
# ----------------------------------------------------------------------------
print("\n=== [1] Reference 5-system ablation (seed 42, LR converged) ===")
gsc = StandardScaler().fit(handcrafted[idx(tr42)])
gfeat_all = gsc.transform(handcrafted)
y_te42 = label[idx(te42)].cpu().numpy()
res = []

# random GCN + style + LR
torch.manual_seed(42)
rnd = GCN(F).to(DEVICE); rnd.eval()
with torch.no_grad():
    e_tr = rnd(node_feat[idx(tr42)], adj[idx(tr42)], mask[idx(tr42)]).cpu().numpy()
    e_te = rnd(node_feat[idx(te42)], adj[idx(te42)], mask[idx(te42)]).cpu().numpy()
Xr_tr = np.hstack([e_tr, gfeat_all[idx(tr42)]])
Xr_te = np.hstack([e_te, gfeat_all[idx(te42)]])
p_rnd = LogisticRegression(max_iter=5000).fit(Xr_tr, label[idx(tr42)].cpu().numpy()).predict(Xr_te)
res.append(("Random GCN + style (untrained)", p_rnd))

# style + LR
p_hc = LogisticRegression(max_iter=5000).fit(gfeat_all[idx(tr42)],
                                             label[idx(tr42)].cpu().numpy()).predict(gfeat_all[idx(te42)])
res.append(("Style descriptors + LR", p_hc))

# FAIR no-graph baseline: per-graph mean-pooled node features + style -> LR.
# Isolates the contribution of graph message-passing: same node + style info,
# no graph convolution. (Node mean-pool is graph-agnostic by construction.)
node_mean = node_feat.mean(dim=1).cpu().numpy()          # [G, F]
scn = StandardScaler().fit(node_mean[idx(tr42)])
Xn_tr = np.hstack([scn.transform(node_mean[idx(tr42)]), gfeat_all[idx(tr42)]])
Xn_te = np.hstack([scn.transform(node_mean[idx(te42)]), gfeat_all[idx(te42)]])
p_node = LogisticRegression(max_iter=5000).fit(
    Xn_tr, label[idx(tr42)].cpu().numpy()).predict(Xn_te)
res.append(("Node mean-pool + style + LR (no graph)", p_node))

# MLP flat
flat = node_feat.reshape(G, -1).cpu().numpy()
scf = StandardScaler().fit(flat[idx(tr42)])
p_mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                      early_stopping=True).fit(scf.transform(flat[idx(tr42)]),
                                               label[idx(tr42)].cpu().numpy()
                                               ).predict(scf.transform(flat[idx(te42)]))
res.append(("MLP (flat raw node feats)", p_mlp))

# trained GCN raw (no style)
p_raw, _, _ = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat=None, use_style=False,
                        train_epochs=60, seed=42)
res.append(("Trained GCN (raw graph only)", p_raw))

# trained GCN + style (ours)
p_ours, emb_ours, model_ours = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat_all,
                                         use_style=True, train_epochs=60, seed=42,
                                         verbose=True)
res.append(("Trained GCN + style (ours)", p_ours))

for nm, p in res:
    print(f"  {nm:32s} acc={accuracy_score(y_te42, p):.3f}  "
          f"macroF1={f1_score(y_te42, p, average='macro'):.3f}")

# McNemar: ours vs each baseline
mcn = []
for nm, p in res:
    if "ours" not in nm:
        mcn.append({"vs": nm, "mcnemar_p": mcnemar_p(y_te42, p_ours, p)})
pd.DataFrame(mcn).to_csv(f"{OUT}/mcnemar.csv", index=False)
print("\nMcNemar (ours vs baseline):")
print(pd.DataFrame(mcn).to_string(index=False))

# ----------------------------------------------------------------------------
# 2. Multi-seed robustness (ours, fresh split + fresh init per seed)
# ----------------------------------------------------------------------------
print("\n=== [2] Multi-seed robustness (Trained GCN + style) ===")
seeds = [42, 7, 13, 99, 2026]
mf1s, accs = [], []
for s in seeds:
    trs, vas, tes = match_split(meta[male_mask], seed=s)
    gsc_s = StandardScaler().fit(handcrafted[idx(trs)])
    gf_s = gsc_s.transform(handcrafted)
    ps, _, _ = train_gcn(idx(trs), idx(vas), idx(tes), gf_s, use_style=True,
                         train_epochs=60, seed=s)
    ys = label[idx(tes)].cpu().numpy()
    a = accuracy_score(ys, ps); f = f1_score(ys, ps, average="macro")
    accs.append(a); mf1s.append(f)
    print(f"  seed {s:5d}  acc={a:.3f}  macroF1={f:.3f}  (n={len(ys)})")
pd.DataFrame([{"metric": "accuracy", "mean": np.mean(accs), "std": np.std(accs)},
              {"metric": "macro_f1", "mean": np.mean(mf1s), "std": np.std(mf1s)}]
             ).to_csv(f"{OUT}/robustness.csv", index=False)
print(f"  macro-F1: {np.mean(mf1s):.3f} +/- {np.std(mf1s):.3f}   "
      f"acc: {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

# ----------------------------------------------------------------------------
# 3. Leave-one-league-out transfer (male clubs) + cross-gender reference
# ----------------------------------------------------------------------------
print("\n=== [3] Cross-league transfer (leave-one-league-out, male) ===")
LEAGUES = ["La Liga", "Ligue 1", "Premier League", "Serie A"]
rows = []
rows.append({"direction": "same-distribution (male test)",
             "n": int(te42.sum()), "accuracy": accuracy_score(y_te42, p_ours),
             "macro_f1": f1_score(y_te42, p_ours, average="macro"),
             "acc_std": 0.0, "f1_std": 0.0, "train_g": int(tr42.sum())})
for L in LEAGUES:
    f1s_l, accs_l = [], []
    test_mask = male_mask & (meta["competition_name"] == L).values
    for s in seeds:
        train_mask = male_mask & ~(meta["competition_name"] == L).values
        # val = 15% of the train matches (stratified), per seed
        tv = meta[train_mask].copy()
        tv["mo"] = np.where(tv["home_score"] > tv["away_score"], "H",
                    np.where(tv["home_score"] < tv["away_score"], "A", "D"))
        uniq = tv[["match_id", "mo"]].drop_duplicates().reset_index(drop=True)
        tr_m, va_m = train_test_split(uniq["match_id"].values, test_size=0.15,
                                      random_state=s, stratify=uniq["mo"].values)
        tr_m2 = train_mask & meta["match_id"].isin(tr_m).values
        va_m2 = train_mask & meta["match_id"].isin(va_m).values
        gsc_l = StandardScaler().fit(handcrafted[idx(tr_m2)])
        gf_l = gsc_l.transform(handcrafted)
        pl, _, _ = train_gcn(idx(tr_m2), idx(va_m2), idx(test_mask), gf_l,
                             use_style=True, train_epochs=60, seed=s)
        yl = label[idx(test_mask)].cpu().numpy()
        accs_l.append(accuracy_score(yl, pl)); f1s_l.append(f1_score(yl, pl, average="macro"))
    rows.append({"direction": f"cross-league -> {L}", "n": int(test_mask.sum()),
                 "accuracy": float(np.mean(accs_l)), "macro_f1": float(np.mean(f1s_l)),
                 "acc_std": float(np.std(accs_l)), "f1_std": float(np.std(f1s_l)),
                 "train_g": int((male_mask & ~(meta["competition_name"] == L).values).sum())})
    print(f"  {L}: macroF1 {np.mean(f1s_l):.3f} +/- {np.std(f1s_l):.3f}  "
          f"(n={test_mask.sum()}, {len(seeds)} seeds)")
# cross-gender reference (male-trained model scored on female graphs)
gcn_m, head_m = model_ours
gcn_m.eval(); head_m.eval()
gxf = torch.tensor(gfeat_all[female_mask], dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    emb_f = gcn_m(node_feat[female_mask], adj[female_mask], mask[female_mask])
    pf = head_m(torch.cat([emb_f, gxf], dim=1)).argmax(1).cpu().numpy()
yf = label[female_mask].cpu().numpy()
rows.append({"direction": "cross-gender -> female", "n": int(female_mask.sum()),
             "accuracy": accuracy_score(yf, pf),
             "macro_f1": f1_score(yf, pf, average="macro"),
             "acc_std": 0.0, "f1_std": 0.0, "train_g": int(tr42.sum())})

trf = pd.DataFrame(rows)
trf.to_csv(f"{OUT}/transfer_league.csv", index=False)
print("\n=== transfer matrix ===")
print(trf.to_string(index=False))

# figure
plt.figure(figsize=(8, 4.5))
trf_sorted = trf.sort_values("macro_f1", ascending=False)
colors = ["#4C72B0" if "same" in d else ("#C44E52" if "female" in d else "#55A868")
          for d in trf_sorted["direction"]]
plt.barh(trf_sorted["direction"], trf_sorted["macro_f1"],
         xerr=trf_sorted["f1_std"], color=colors, capsize=3)
for i, (v, n) in enumerate(zip(trf_sorted["macro_f1"], trf_sorted["n"])):
    plt.text(v + 0.004, i, f"{v:.3f} (n={n})", va="center", fontsize=8)
plt.xlabel("macro-F1")
plt.title("GCN transfer — same distribution vs cross-league vs cross-gender")
plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig(f"{OUT}/fig_transfer_compare.png", dpi=130); plt.close()
print(f"\n[saved] transfer_league.csv, robustness.csv, mcnemar.csv, "
      f"fig_transfer_compare.png")
