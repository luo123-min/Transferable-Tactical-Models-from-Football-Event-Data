"""
supplementary_analysis.py — Supplementary robustness and validation experiments.

Produces (results_gnn/):
  ablation_8systems.csv     8-system ablation (6 original + fair + naive)
  mcnemar_tests.csv         exact McNemar p (ours vs fair / graph-free / naive)
  leave_teams_out.csv       male test split by team novelty
  transfer_significance.csv same-distribution vs each transfer target (z on means)

Keeps make_figures.py / transfer_league.py untouched so the published figure
pipeline is not disturbed; the author can later merge these baselines in.
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
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from scipy.stats import binom

male_mask = (meta["competition_gender"] == "male").values
female_mask = ~male_mask
tr42, va42, te42 = match_split(meta[male_mask], seed=42)
gsc = StandardScaler().fit(handcrafted[idx(tr42)])
gfeat_all = gsc.transform(handcrafted)
y_te42 = label[idx(te42)].cpu().numpy()

print("\n=== 8-system ablation (seed 42) ===")
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

p_raw = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat=None, use_style=False, seed=42)
rows.append(("Trained GCN (raw graph only)", p_raw))

# Representation-matched control: same architecture & 406-d
# input, but IDENTITY adjacency => no message passing (per-node MLP + identical
# readout). Isolates the contribution of graph convolution.
adj_eye = torch.eye(node_feat.shape[1], device=DEVICE).unsqueeze(0).repeat(G, 1, 1)
p_fair = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat_all, use_style=True,
                  seed=42, adj_override=adj_eye)
rows.append(("Node-MLP + pool + style (no message passing)", p_fair))

# Naive baseline: majority class from training distribution.
maj = np.bincount(y_te42).argmax()
p_maj = np.full_like(y_te42, maj)
rows.append(("Majority-class prior (train)", p_maj))

p_ours = train_gcn(idx(tr42), idx(va42), idx(te42), gfeat_all, use_style=True, seed=42)
rows.append(("Trained GCN + style (ours)", p_ours))

abl = pd.DataFrame([{"model": nm, "accuracy": accuracy_score(y_te42, p),
                     "macro_f1": f1_score(y_te42, p, average="macro")}
                    for nm, p in rows]).sort_values("macro_f1")
abl.to_csv(f"{OUT}/ablation_8systems.csv", index=False)
print(abl.to_string(index=False))

# ---- McNemar (exact binomial) ours vs each baseline -----------------------
def mcnemar_p(y, a, b):
    b_ = int(np.sum((a == y) & (b != y))); c_ = int(np.sum((a != y) & (b == y)))
    n_ = b_ + c_
    return 1.0 if n_ == 0 else min(1.0, binom.cdf(min(b_, c_), n_, 0.5) * 2.0)

preds = {nm: p for nm, p in rows}
mc = []
for nm in ["Node mean-pool + style + LR (no graph)",
           "Node-MLP + pool + style (no message passing)",
           "Majority-class prior (train)"]:
    mc.append({"vs": nm, "mcnemar_p": mcnemar_p(y_te42, p_ours, preds[nm])})
pd.DataFrame(mc).to_csv(f"{OUT}/mcnemar_tests.csv", index=False)
print("\nMcNemar (ours vs baseline):")
print(pd.DataFrame(mc).to_string(index=False))

# ---- Leave-teams-out novelty check --------------------------
print("\n=== leave-teams-out novelty check (seed 42 male test) ===")
tr_teams = set(meta.iloc[idx(tr42)]["team_id"].unique())
te_idx = idx(te42)
teams_te = meta.iloc[te_idx]["team_id"].values
seen = np.array([t in tr_teams for t in teams_te])
lt = []
for name, m in [("teams seen in training", seen), ("teams held out", ~seen)]:
    sub = te_idx[m]
    if len(sub) == 0:
        continue
    mf1 = f1_score(label[sub].cpu().numpy(), p_ours[m], average="macro")
    lt.append({"group": name, "n": int(len(sub)),
               "macro_f1": round(float(mf1), 4)})
    print(f"  {name:24s} n={len(sub):4d}  macroF1={mf1:.3f}")
pd.DataFrame(lt).to_csv(f"{OUT}/leave_teams_out.csv", index=False)

# ---- Transfer significance: same-distribution 5 seeds vs each
# transfer target's 5 seeds (z on the means using standard errors). ----------
print("\n=== transfer significance (same-dist vs transfer targets) ===")
try:
    perseed = pd.read_csv("figures_paper/robustness_perseed.csv")
    same_f1 = perseed["macro_f1"].values.astype(float)
    same_mean = float(np.mean(same_f1)); se_same = float(np.std(same_f1)) / np.sqrt(len(same_f1))
    trf = pd.read_csv(f"{OUT}/transfer_league.csv")
    ts = []
    for _, r in trf.iterrows():
        if str(r["direction"]).startswith("same"):
            continue
        tm = float(r["macro_f1"]); se_t = float(r["f1_std"]) / np.sqrt(5)
        z = (tm - same_mean) / np.sqrt(se_same**2 + se_t**2)
        ts.append({"target": r["direction"], "macro_f1": round(tm, 4),
                   "delta": round(tm - same_mean, 4), "z": round(float(z), 2)})
        print(f"  {r['direction']:30s} macroF1={tm:.3f}  Δ={tm-same_mean:+.3f}  z={z:.2f}")
    pd.DataFrame(ts).to_csv(f"{OUT}/transfer_significance.csv", index=False)
except Exception as e:
    print("  (skipped: need figures_paper/robustness_perseed.csv + transfer_league.csv)", e)

print("\n[done] see results_gnn/ (ablation_8systems.csv, mcnemar_tests.csv, leave_teams_out.csv, transfer_significance.csv)")
