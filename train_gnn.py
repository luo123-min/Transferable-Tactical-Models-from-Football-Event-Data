"""
train_gnn.py — Trained Graph Neural Network for football tactical modeling.

Replaces the untrained / random GCN in the earlier prototype with a GCN that
is actually trained end-to-end on the full StatsBomb passing-network corpus to
predict each team's match result from its passing graph.

Compared systems (all on the SAME per-team-match label grain):
  1. Random (untrained) GCN  -> frozen random embeddings + logistic head
  2. Hand-crafted + LogisticRegression   (classic ML, no deep learning)
  3. MLP on flattened node features       (deep, but NO graph structure)
  4. Trained GCN  (ours)                  (deep + message passing)

Outputs (results_gnn/):
  baseline_results.csv
  fig_confusion_gnn.png
  fig_baseline_compare.png
  fig_embedding_pca.png
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
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             confusion_matrix, normalized_mutual_info_score)

torch.manual_seed(42)
np.random.seed(42)

DATA = "results_gnn/graph_data.npz"
META = "results_gnn/graph_meta.csv"
OUT = "results_gnn"
os.makedirs(OUT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[env] device={DEVICE}  torch={torch.__version__}")

# ----------------------------------------------------------------------------
# Load data
# ----------------------------------------------------------------------------
D = np.load(DATA)
node_feat = torch.tensor(D["node_feat"], dtype=torch.float32, device=DEVICE)
adj = torch.tensor(D["adj"], dtype=torch.float32, device=DEVICE)
mask = torch.tensor(D["mask"], dtype=torch.float32, device=DEVICE)
label = torch.tensor(D["label"], dtype=torch.long, device=DEVICE)
handcrafted = D["handcrafted"].astype(np.float32)
meta = pd.read_csv(META)
G = node_feat.shape[0]
F = node_feat.shape[2]
print(f"[data] graphs={G}  node_feat={tuple(node_feat.shape)}  "
      f"label dist={dict(zip(*np.unique(D['label'], return_counts=True)))}")

# ----------------------------------------------------------------------------
# Leak-free split by MATCH — MALE-only train/val/test; FEMALE = transfer set
# ----------------------------------------------------------------------------
meta["match_outcome"] = np.where(meta["home_score"] > meta["away_score"], "H",
                          np.where(meta["home_score"] < meta["away_score"], "A", "D"))
male_mask = (meta["competition_gender"] == "male").values
xf = ~male_mask                      # cross-gender transfer set (all female graphs)

uniq_m = meta[male_mask][["match_id", "match_outcome"]].drop_duplicates().reset_index(drop=True)
mids = uniq_m["match_id"].values
mout = uniq_m["match_outcome"].values
mid_tr, mid_tmp, mout_tr, mout_tmp = train_test_split(
    mids, mout, test_size=0.30, random_state=42, stratify=mout)
mid_val, mid_te = train_test_split(
    mid_tmp, test_size=0.50, random_state=42, stratify=mout_tmp)

m_tr = meta["match_id"].isin(mid_tr).values
m_va = meta["match_id"].isin(mid_val).values
m_te = meta["match_id"].isin(mid_te).values
tr = male_mask & m_tr
va = male_mask & m_va
te = male_mask & m_te
print(f"[split] MALE train={tr.sum()} val={va.sum()} test={te.sum()} graphs "
      f"| FEMALE transfer={xf.sum()} graphs")

def idx(a):
    return np.where(a)[0]

# ----------------------------------------------------------------------------
# GCN model
# ----------------------------------------------------------------------------
class GCN(nn.Module):
    def __init__(self, f_in, hidden=128, n_layers=2, n_class=3, head=64, dropout=0.3):
        super().__init__()
        self.Win = nn.Linear(f_in, hidden)
        self.convs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head1 = nn.Linear(hidden * 3, head)   # mean + max + sum pool
        self.head2 = nn.Linear(head, n_class)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj, mask):
        h = torch.relu(self.Win(x))
        for conv, bn in zip(self.convs, self.bns):
            support = torch.bmm(adj, h)             # [B, N, hidden]
            h2 = torch.relu(bn(conv(support)))
            h = h + h2                              # residual
        mb = mask.unsqueeze(-1)
        hm = h * mb
        meanp = hm.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        maxp = (hm + (1.0 - mb) * (-1e9)).max(1).values
        sump = hm.sum(1)
        pooled = torch.cat([meanp, maxp, sump], dim=1)
        return pooled


def run_gcn(train_idx, val_idx, test_idx, gfeat=None, use_style=True, train_epochs=120):
    from sklearn.utils.class_weight import compute_class_weight
    ytr = label[train_idx].cpu().numpy()
    cls_w = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=ytr)
    cw = torch.tensor(cls_w, dtype=torch.float32, device=DEVICE)
    crit = nn.CrossEntropyLoss(weight=cw)

    gcn = GCN(F, hidden=128, n_layers=2).to(DEVICE)
    pool_dim = 128 * 3
    g_dim = gfeat.shape[1] if (use_style and gfeat is not None) else 0
    head = nn.Sequential(
        nn.Linear(pool_dim + g_dim, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 3),
    ).to(DEVICE)
    opt = torch.optim.Adam(list(gcn.parameters()) + list(head.parameters()),
                           lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5,
                                                       patience=8, min_lr=1e-5)
    best_val, best_state, patience = -1, None, 0
    if use_style and gfeat is not None:
        gt = torch.tensor(gfeat[train_idx], dtype=torch.float32, device=DEVICE)
        gv = torch.tensor(gfeat[val_idx], dtype=torch.float32, device=DEVICE)
        ge = torch.tensor(gfeat[test_idx], dtype=torch.float32, device=DEVICE)
        tr_b = TensorDataset(node_feat[train_idx], adj[train_idx], mask[train_idx], gt, label[train_idx])
    else:
        tr_b = TensorDataset(node_feat[train_idx], adj[train_idx], mask[train_idx], label[train_idx])
    dl = DataLoader(tr_b, batch_size=128, shuffle=True)

    def head_forward(pooled, g=None):
        """pooled: [B, pool_dim]; g: [B, g_dim] (batch) or None."""
        if use_style and g is not None:
            return head(torch.cat([pooled, g], dim=1))
        return head(pooled)

    # full-split global feats for eval (val/test pass whole split at once)
    g_full = {"train": gt, "val": gv, "test": ge} if (use_style and gfeat is not None) else {}

    for ep in range(train_epochs):
        gcn.train(); head.train()
        for batch in dl:
            if use_style:
                xb, ab, mb, gxb, lb = batch
            else:
                xb, ab, mb, lb = batch
                gxb = None
            opt.zero_grad()
            pooled = gcn(xb, ab, mb)
            out = head_forward(pooled, gxb)            # batch global feats (None if no-style)
            loss = crit(out, lb)
            loss.backward(); opt.step()
        gcn.eval(); head.eval()
        with torch.no_grad():
            pv_p = gcn(node_feat[val_idx], adj[val_idx], mask[val_idx])
            pv = head_forward(pv_p, g_full.get("val"))
            vp = pv.argmax(1).cpu().numpy()
        vf1 = f1_score(label[val_idx].cpu().numpy(), vp, average="macro")
        if ep % 10 == 0 or ep == train_epochs - 1:
            vacc = accuracy_score(label[val_idx].cpu().numpy(), vp)
            print(f"    ep {ep:3d}  loss={loss.item():.3f}  val_acc={vacc:.3f}  val_macroF1={vf1:.3f}")
        sched.step(vf1)
        if vf1 > best_val + 1e-4:
            best_val = vf1
            best_state = (dict(gcn.named_parameters()), dict(head.named_parameters()))
            patience = 0
        else:
            patience += 1
            if patience >= 20:
                break
    sd_g = {k: v for k, v in best_state[0].items()}
    sd_h = {k: v for k, v in best_state[1].items()}
    gcn.load_state_dict(sd_g); head.load_state_dict(sd_h)
    gcn.eval(); head.eval()
    with torch.no_grad():
        pe_p = gcn(node_feat[test_idx], adj[test_idx], mask[test_idx])
        pe = head_forward(pe_p, g_full.get("test"))
        tp = pe.argmax(1).cpu().numpy()
        te_emb = gcn(node_feat[test_idx], adj[test_idx], mask[test_idx]).detach().cpu().numpy()
        tr_emb = gcn(node_feat[train_idx], adj[train_idx], mask[train_idx]).detach().cpu().numpy()
    return tp, te_emb, tr_emb, (gcn, head)


# ----------------------------------------------------------------------------
# Baselines
# ----------------------------------------------------------------------------
def eval_pred(y_true, y_pred, name):
    acc = accuracy_score(y_true, y_pred)
    mf1 = f1_score(y_true, y_pred, average="macro")
    print(f"  {name:32s} acc={acc:.3f}  macroF1={mf1:.3f}")
    return {"model": name, "accuracy": acc, "macro_f1": mf1}


y_test = label[te].cpu().numpy()

# global style descriptors (the same rich features the classic ML baseline uses),
# standardised on the training split so the GNN and the LR see identical inputs.
gsc = StandardScaler().fit(handcrafted[idx(tr)])
gfeat_all = gsc.transform(handcrafted)          # [G, g_dim], full corpus
results = []

# 1) Random (untrained) GCN structure emb + style descriptors + LR head
print("\n[baseline] 1. Random (untrained) GCN + style + LR head")
rnd = GCN(F).to(DEVICE)
rnd.eval()
with torch.no_grad():
    e_tr = rnd(node_feat[idx(tr)], adj[idx(tr)], mask[idx(tr)]).cpu().numpy()
    e_te = rnd(node_feat[idx(te)], adj[idx(te)], mask[idx(te)]).cpu().numpy()
Xr_tr = np.hstack([e_tr, gfeat_all[idx(tr)]]); Xr_te = np.hstack([e_te, gfeat_all[idx(te)]])
lr = LogisticRegression(max_iter=5000).fit(Xr_tr, label[idx(tr)].cpu().numpy())
p_rnd = lr.predict(Xr_te)
results.append(eval_pred(y_test, p_rnd, "Random GCN + style (untrained)"))

# 2) Style descriptors + LogisticRegression (classic ML, no deep learning)
print("\n[baseline] 2. Hand-crafted style + LogisticRegression")
hc_tr = gfeat_all[idx(tr)]; hc_te = gfeat_all[idx(te)]
lr2 = LogisticRegression(max_iter=5000).fit(hc_tr, label[idx(tr)].cpu().numpy())
p_hc = lr2.predict(hc_te)
results.append(eval_pred(y_test, p_hc, "Style descriptors + LR"))

# 3) MLP on flattened raw node features (deep, but NO graph structure / NO style)
print("\n[baseline] 3. MLP on flattened raw node features")
flat = node_feat.reshape(G, -1).cpu().numpy()
scf = StandardScaler().fit(flat[idx(tr)])
fl_tr = scf.transform(flat[idx(tr)]); fl_te = scf.transform(flat[idx(te)])
mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, random_state=42,
                    early_stopping=True)
mlp.fit(fl_tr, label[idx(tr)].cpu().numpy())
p_mlp = mlp.predict(fl_te)
results.append(eval_pred(y_test, p_mlp, "MLP (flat raw node feats)"))

# 4) Trained GCN on RAW graph only (no style descriptors) — structure ablation
print("\n[model] 4a. Trained GCN (raw graph only, no style)")
p_gnn_raw, emb_te_raw, emb_tr_raw, _ = run_gcn(idx(tr), idx(va), idx(te), gfeat=None,
                                              use_style=False, train_epochs=60)
results.append(eval_pred(y_test, p_gnn_raw, "Trained GCN (raw graph only)"))

# 5) Trained GCN (learned structure emb) + style descriptors
print("\n[model] 5. Trained GCN + style descriptors")
p_gnn, emb_te_gnn, emb_tr_gnn, gnn = run_gcn(idx(tr), idx(va), idx(te), gfeat_all, train_epochs=60)
results.append(eval_pred(y_test, p_gnn, "Trained GCN + style (ours)"))

# ----------------------------------------------------------------------------
# Cross-gender transfer: female graphs scored by the male-trained model
# (no fine-tuning; the same scaler fitted on male train is applied to female)
# ----------------------------------------------------------------------------
print("\n=== Cross-gender transfer (female graphs, male-trained model, no tuning) ===")
gcn_m, head_m = gnn
gcn_m.eval(); head_m.eval()
gxf_t = torch.tensor(gfeat_all[xf], dtype=torch.float32, device=DEVICE)
with torch.no_grad():
    xf_emb = gcn_m(node_feat[xf], adj[xf], mask[xf]).detach().cpu().numpy()
    pe_xf = head_m(torch.cat([torch.tensor(xf_emb, device=DEVICE), gxf_t], dim=1))
    p_xf = pe_xf.argmax(1).cpu().numpy()
y_xf = label[xf].cpu().numpy()
xf_acc = accuracy_score(y_xf, p_xf)
xf_f1 = f1_score(y_xf, p_xf, average="macro")
ref_f1 = f1_score(y_test, p_gnn, average="macro")
ref_acc = accuracy_score(y_test, p_gnn)
print(f"  FEMALE transfer  acc={xf_acc:.3f}  macroF1={xf_f1:.3f}  (n={len(y_xf)})")
print(f"  MALE test (ref)  acc={ref_acc:.3f}  macroF1={ref_f1:.3f}  (n={len(y_test)})")
pd.DataFrame([{"model": "Cross-gender transfer (male model -> female graphs)",
               "accuracy": xf_acc, "macro_f1": xf_f1, "n": len(y_xf)},
              {"model": "Male test (reference)", "accuracy": ref_acc,
               "macro_f1": ref_f1, "n": len(y_test)}]).to_csv(
    f"{OUT}/cross_gender_transfer.csv", index=False)

# ----------------------------------------------------------------------------
# Detailed report + figures for the trained GCN
# ----------------------------------------------------------------------------
print("\n=== Trained GCN — classification report (test) ===")
print(classification_report(y_test, p_gnn, target_names=["loss", "draw", "win"], digits=3))

cm = confusion_matrix(y_test, p_gnn)
plt.figure(figsize=(5, 4))
plt.imshow(cm, cmap="Blues")
plt.title("Trained GCN — confusion matrix (test)")
plt.xticks([0, 1, 2], ["loss", "draw", "win"]); plt.yticks([0, 1, 2], ["loss", "draw", "win"])
plt.xlabel("predicted"); plt.ylabel("true")
for i in range(3):
    for j in range(3):
        plt.text(j, i, cm[i, j], ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.tight_layout(); plt.savefig(f"{OUT}/fig_confusion_gnn.png", dpi=130); plt.close()

# baseline comparison
rdf = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
print("\n=== SUMMARY (test) ===")
print(rdf.to_string(index=False))
rdf.to_csv(f"{OUT}/baseline_results.csv", index=False)

plt.figure(figsize=(7, 4))
colors = ["#888" if "Random" in m else ("#4C72B0" if "Trained" in m else "#55A868")
          for m in rdf["model"]]
plt.barh(rdf["model"], rdf["macro_f1"], color=colors)
plt.xlabel("test macro-F1 (3-class result)")
plt.title("Tactical outcome prediction — model comparison")
plt.gca().invert_yaxis(); plt.tight_layout()
plt.savefig(f"{OUT}/fig_baseline_compare.png", dpi=130); plt.close()

# embedding PCA — trained GCN: male test + female transfer in ONE shared space
from sklearn.decomposition import PCA
emb_all = np.vstack([emb_tr_gnn, emb_te_gnn, xf_emb])
pca = PCA(n_components=2).fit(emb_all)
emb2 = pca.transform(emb_te_gnn)
emb2f = pca.transform(xf_emb)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for v, c, nm in zip([0, 1, 2], ["#C44E52", "#DD8452", "#55A868"],
                    ["loss", "draw", "win"]):
    m = (y_test == v)
    axes[0].scatter(emb2[m, 0], emb2[m, 1], s=8, alpha=0.5, color=c, label=nm)
axes[0].set_title("GNN embedding PCA — male test by true result")
axes[0].set_xlabel("PC1"); axes[0].set_ylabel("PC2"); axes[0].legend(fontsize=8)
axes[1].scatter(emb2[:, 0], emb2[:, 1], s=8, alpha=0.5, color="#4C72B0", label="male (test)")
axes[1].scatter(emb2f[:, 0], emb2f[:, 1], s=8, alpha=0.5, color="#C44E52", label="female (transfer)")
axes[1].set_title("GNN embedding PCA — male vs female")
axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2"); axes[1].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/fig_embedding_pca.png", dpi=130); plt.close()

print("\n[saved] baseline_results.csv, fig_confusion_gnn.png, "
      "fig_baseline_compare.png, fig_embedding_pca.png")
