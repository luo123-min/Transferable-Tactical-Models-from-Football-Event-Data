"""
gnn_common.py — Shared graph-data loading and GCN training utilities.

Single source of truth for the duplicated definitions that used to live in
both supplementary_analysis.py and control_baseline_seeds.py:
  - data loading (results_gnn/graph_data.npz + graph_meta.csv)
  - idx(), match_split()   (match-level stratified train/val/test split)
  - GCN                    (graph convolution + readout)
  - train_gcn()            (training loop + best-macro-F1 checkpoint)

Both scripts import from here so the model code lives in exactly one place.
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

DATA = "results_gnn/graph_data.npz"
META = "results_gnn/graph_meta.csv"
OUT = "results_gnn"
os.makedirs(OUT, exist_ok=True)
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
    m2 = male_df.copy()
    m2["mo"] = np.where(m2["home_score"] > m2["away_score"], "H",
                np.where(m2["home_score"] < m2["away_score"], "A", "D"))
    uniq = m2[["match_id", "mo"]].drop_duplicates().reset_index(drop=True)
    tr, tmp = train_test_split(uniq["match_id"].values, test_size=test_frac,
                               random_state=seed, stratify=uniq["mo"].values)
    tmp_u = uniq[uniq["match_id"].isin(tmp)]
    va, te = train_test_split(tmp_u["match_id"].values, test_size=0.5,
                              random_state=seed, stratify=tmp_u["mo"].values)
    return (m2["match_id"].isin(tr).values,
            m2["match_id"].isin(va).values,
            m2["match_id"].isin(te).values)


class GCN(nn.Module):
    def __init__(self, f_in, hidden=128, n_layers=2, n_class=3, head=64, dropout=0.3):
        super().__init__()
        self.Win = nn.Linear(f_in, hidden)
        self.convs = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.bns = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.head1 = nn.Linear(hidden * 3, head)
        self.head2 = nn.Linear(head, n_class)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, adj_in, mask):
        h = torch.relu(self.Win(x))
        for conv, bn in zip(self.convs, self.bns):
            support = torch.bmm(adj_in, h)
            h2 = torch.relu(bn(conv(support)))
            h = h + h2
        mb = mask.unsqueeze(-1)
        hm = h * mb
        meanp = hm.sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
        maxp = (hm + (1.0 - mb) * (-1e9)).max(1).values
        sump = hm.sum(1)
        return torch.cat([meanp, maxp, sump], dim=1)


def train_gcn(train_idx, val_idx, test_idx, gfeat=None, use_style=True,
              train_epochs=60, seed=42, verbose=False, adj_override=None):
    torch.manual_seed(seed); np.random.seed(seed)
    Ad = adj_override if adj_override is not None else adj
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
        tr_b = TensorDataset(node_feat[train_idx], Ad[train_idx], mask[train_idx],
                             gt, label[train_idx])
    else:
        gv = ge = None
        tr_b = TensorDataset(node_feat[train_idx], Ad[train_idx], mask[train_idx],
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
            pv = hf(gcn(node_feat[val_idx], Ad[val_idx], mask[val_idx]), gv)
            vp = pv.argmax(1).cpu().numpy()
        vf1 = f1_score(label[val_idx].cpu().numpy(), vp, average="macro")
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
        pe = hf(gcn(node_feat[test_idx], Ad[test_idx], mask[test_idx]), ge)
        tp = pe.argmax(1).cpu().numpy()
    return tp
