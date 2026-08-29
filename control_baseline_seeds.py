"""
control_baseline_seeds.py — 5-seed paired confirmation of the representation-matched control.

The representation-matched control uses an identity-adjacency MLP (same 406-d
input, no message passing). The seed-42 run showed it scores 0.530 vs the
GCN's 0.522 (McNemar p=0.63, not significant). Here we confirm across the
SAME 5 seeds used by the published robustness table ([42,7,13,99,2026])
so the comparison is strictly paired.
"""

# Shared data loading, GCN, train_gcn and match_split live in gnn_common so
# they are defined exactly once.
from gnn_common import (
    node_feat, adj, mask, label, handcrafted, meta, G, F, DEVICE, OUT,
    idx, match_split, GCN, train_gcn,
    StandardScaler, accuracy_score, f1_score,
)
import numpy as np
import pandas as pd
import torch
from scipy.stats import ttest_rel, wilcoxon

male_mask = (meta["competition_gender"] == "male").values
SEEDS = [42, 7, 13, 99, 2026]
adj_eye = torch.eye(node_feat.shape[1], device=DEVICE).unsqueeze(0).repeat(G, 1, 1)
rows = []
for s in SEEDS:
    tr, va, te = match_split(meta[male_mask], seed=s)
    # Fit the style-descriptor scaler on the TRAINING split only (no leakage),
    # matching supplementary_analysis.py so the seed-42 point reproduces tab:e3.
    gsc = StandardScaler().fit(handcrafted[idx(tr)])
    gfeat_all = gsc.transform(handcrafted)
    p_ours = train_gcn(idx(tr), idx(va), idx(te), gfeat_all, True, seed=s)
    p_fair = train_gcn(idx(tr), idx(va), idx(te), gfeat_all, True, seed=s, adj_override=adj_eye)
    y = label[idx(te)].cpu().numpy()
    rows.append({"seed": s,
                 "ours_acc": round(accuracy_score(y, p_ours), 4),
                 "ours_mf1": round(f1_score(y, p_ours, average="macro"), 4),
                 "fair_acc": round(accuracy_score(y, p_fair), 4),
                 "fair_mf1": round(f1_score(y, p_fair, average="macro"), 4)})
    print(f"seed {s}: ours={rows[-1]['ours_mf1']:.4f}  fair={rows[-1]['fair_mf1']:.4f}")
df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/control_seeds.csv", index=False)

ours = df["ours_mf1"].values; fair = df["fair_mf1"].values
print("\n=== 5-seed paired summary (ours vs fair control) ===")
print(f"ours : {ours.mean():.4f} ± {ours.std():.4f}")
print(f"fair : {fair.mean():.4f} ± {fair.std():.4f}")
print(f"mean Δ (ours-fair) = {ours.mean()-fair.mean():+.4f}")
print(f"paired t-test   p = {ttest_rel(ours,fair).pvalue:.4f}")
try:
    print(f"Wilcoxon signed-rank p = {wilcoxon(ours,fair).pvalue:.4f}")
except Exception as e:
    print("wilcoxon skipped:", e)
print("\n[done] see results_gnn/control_seeds.csv")
