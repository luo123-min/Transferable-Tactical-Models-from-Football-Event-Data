"""
Topic 3 prototype: AI-driven football tactical modeling from PUBLIC StatsBomb event data.
Pipeline:
  1. Build per-team aggregated passing networks (nodes=players, directed weighted edges=passes).
  2. Compute graph-level + spatial passing-style features per team.
  3. GCN (numpy, seeded, unsupervised structural encoder) -> graph embeddings.
  4. Combine features + embeddings -> clustering (KMeans/GMM) -> tactical prototypes.
  5. Interpretability: RandomForest feature importances + cluster centroid profiles.
  6. Visualizations: passing networks, PCA scatter, centroid heatmap, importance bars.
  7. Downstream demo: predict match result from team embeddings (tactical -> outcome).
Outputs: results/*.csv, results/clusters.json, figures/*.png
"""
import json, os, math, warnings
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
warnings.filterwarnings("ignore")

BASE = os.path.dirname(__file__)
DATA = os.path.join(BASE, "data")
EVT_DIR = os.path.join(DATA, "events")
FIG = os.path.join(BASE, "figures")
RES = os.path.join(BASE, "results")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

COMP, SEASON = 43, 3

# ---------- load matches for team names + results ----------
with open(os.path.join(DATA, f"matches_{COMP}_{SEASON}.json")) as f:
    matches = json.load(f)
team_name = {}
match_results = {}  # match_id -> (home_id, away_id, label 0/1/2: home/draw/away)
for m in matches:
    h, a = m["home_team"], m["away_team"]
    hid, hname = h["home_team_id"], h["home_team_name"]
    aid, aname = a["away_team_id"], a["away_team_name"]
    team_name[hid] = hname
    team_name[aid] = aname
    hs, ast = m.get("home_score"), m.get("away_score")
    if hs is None or ast is None:
        continue
    label = 0 if hs > ast else (2 if ast > hs else 1)
    match_results[m["match_id"]] = (hid, aid, label)

# ---------- parse events: per-team passes ----------
# team_id -> list of pass records: (passer_id, passer_pos, recip_id, recip_pos, start_x, start_y, end_x, end_y, complete)
team_passes = {}
files = sorted(os.listdir(EVT_DIR))
for fn in files:
    if not fn.endswith(".json"):
        continue
    try:
        with open(os.path.join(EVT_DIR, fn)) as f:
            evts = json.load(f)
    except Exception as e:
        print("  skip bad file", fn, e)
        continue
    mid = int(fn.replace(".json", ""))
    for e in evts:
        if e.get("type", {}).get("id") != 30:  # pass
            continue
        team = e.get("team", {}).get("id")
        if team is None:
            continue
        passer = e.get("player", {})
        pid = passer.get("id")
        ppos = passer.get("position", {}).get("name", "Unknown")
        loc = e.get("location")
        pas = e.get("pass", {})
        recip = pas.get("recipient", {})
        rid = recip.get("id")
        rpos = recip.get("position", {}).get("name", "Unknown")
        end = pas.get("end_location")
        if loc is None or end is None or rid is None:
            continue
        rec = (pid, ppos, rid, rpos, loc[0], loc[1], end[0], end[1], True)
        team_passes.setdefault(team, []).append(rec)
        # also store incomplete passes (no recipient) for completion rate
        if rid is None:
            team_passes[team].append((pid, ppos, None, None, loc[0], loc[1], None, None, False))

print(f"parsed {len(team_passes)} teams' pass data")

# ---------- build passing networks + features ----------
def gcn_embedding(adj, node_feats, seed=0, hidden=16, layers=2):
    np.random.seed(seed)
    n = adj.shape[0]
    A = adj + np.eye(n)
    d = A.sum(1)
    Dinv = np.diag(1.0 / np.sqrt(d))
    Ahat = Dinv @ A @ Dinv
    H = node_feats.astype(float)
    for _ in range(layers):
        W = np.random.randn(H.shape[1], hidden) * 0.6
        H = np.maximum(0.0, Ahat @ H @ W)
    return H.mean(0)  # graph-level embedding

POS_GROUPS = {"GK": "GK", "RB": "DEF", "LB": "DEF", "RCB": "DEF", "LCB": "DEF",
              "CB": "DEF", "RWB": "DEF", "LWB": "DEF", "RDM": "MID", "LDM": "MID",
              "RCM": "MID", "LCM": "MID", "CM": "MID", "RM": "MID", "LM": "MID",
              "RAM": "MID", "LAM": "MID", "CAM": "MID", "CDM": "MID", "RW": "FWD",
              "LW": "FWD", "RF": "FWD", "LF": "FWD", "ST": "FWD", "CF": "FWD",
              "RS": "FWD", "LS": "FWD", "SS": "FWD"}

def build_features(team_id):
    recs = team_passes[team_id]
    comp = [r for r in recs if r[8]]
    att = [r for r in recs if not r[8]]
    total_att = len(recs)
    n_comp = len(comp)
    if n_comp < 5:
        return None
    # nodes = players appearing (passer or recipient) in completed passes
    players = {}
    for r in comp:
        players.setdefault(r[0], []).append(r[1])
        players.setdefault(r[2], []).append(r[3])
    # position per player = most frequent
    pos_of = {}
    for pid, plist in players.items():
        from collections import Counter
        pos_of[pid] = Counter(plist).most_common(1)[0][0]
    pid_list = list(players.keys())
    idx = {p: i for i, p in enumerate(pid_list)}
    n = len(pid_list)
    # directed weighted adjacency (completed passes)
    A = np.zeros((n, n))
    for r in comp:
        i, j = idx[r[0]], idx[r[2]]
        A[i, j] += 1
    G = nx.from_numpy_array(A, create_using=nx.DiGraph)
    Gu = nx.from_numpy_array((A + A.T > 0).astype(float))
    # weights for undirected metrics: use sum of both directions
    W = (A + A.T)
    Guw = nx.from_numpy_array(W)
    feats = {}
    feats["n_players"] = n
    feats["n_passes_completed"] = n_comp
    feats["pass_attempts"] = total_att
    feats["completion_rate"] = n_comp / max(1, total_att)
    # density (directed)
    feats["density_dir"] = nx.density(G)
    # reciprocity
    feats["reciprocity"] = nx.reciprocity(G) if G.number_of_edges() > 0 else 0.0
    # degree stats
    outdeg = np.array([G.out_degree(i, weight="weight") for i in range(n)])
    indeg = np.array([G.in_degree(i, weight="weight") for i in range(n)])
    feats["mean_out_deg"] = outdeg.mean()
    feats["mean_in_deg"] = indeg.mean()
    # undirected metrics
    try:
        feats["global_clustering"] = nx.transitivity(Guw)
    except Exception:
        feats["global_clustering"] = 0.0
    try:
        feats["assortativity"] = nx.degree_assortativity_coefficient(Guw)
    except Exception:
        feats["assortativity"] = 0.0
    try:
        comm = nx.community.greedy_modularity_communities(Guw, weight="weight")
        feats["n_communities"] = len(comm)
        feats["modularity"] = nx.community.modularity(Guw, comm, weight="weight")
    except Exception:
        feats["n_communities"] = 1
        feats["modularity"] = 0.0
    # centrality
    try:
        dc = np.array(list(nx.degree_centrality(Guw).values()))
        feats["mean_deg_cent"] = dc.mean(); feats["max_deg_cent"] = dc.max()
    except Exception:
        feats["mean_deg_cent"] = 0; feats["max_deg_cent"] = 0
    try:
        bc = np.array(list(nx.betweenness_centrality(Guw, weight="weight").values()))
        feats["mean_betw"] = bc.mean(); feats["max_betw"] = bc.max()
    except Exception:
        feats["mean_betw"] = 0; feats["max_betw"] = 0
    try:
        cc = np.array(list(nx.closeness_centrality(Guw).values()))
        feats["mean_close"] = cc.mean(); feats["max_close"] = cc.max()
    except Exception:
        feats["mean_close"] = 0; feats["max_close"] = 0
    try:
        ec = np.array(list(nx.eigenvector_centrality_numpy(Guw, weight="weight").values()))
        feats["mean_eigen"] = ec.mean(); feats["max_eigen"] = ec.max()
    except Exception:
        feats["mean_eigen"] = 0; feats["max_eigen"] = 0
    # spectral
    try:
        L = nx.normalized_laplacian_matrix(Guw).astype(float).todense()
        eigs = np.linalg.eigvalsh(L)
        feats["algebraic_conn"] = float(sorted(eigs)[1]) if n > 1 else 0.0
        feats["spectral_gap"] = float(eigs[-1] - eigs[-2]) if n > 1 else 0.0
    except Exception:
        feats["algebraic_conn"] = 0.0; feats["spectral_gap"] = 0.0
    # spatial passing-style features
    sx, sy, ex, ey = [], [], [], []
    lengths, fwd = [], []
    # forward direction: flip so mean progression positive
    prog = [ (r[6]-r[4]) for r in comp ]
    flip = -1 if np.mean(prog) < 0 else 1
    for r in comp:
        sxp, syp, exp_, eyp = r[4], r[5], r[6], r[7]
        dx = (exp_ - sxp) * flip
        dy = eyp - syp
        length = math.hypot(exp_ - sxp, dy)
        lengths.append(length)
        fwd.append(1 if dx > 5 else 0)
        sx.append(sxp); sy.append(syp); ex.append(exp_); ey.append(eyp)
    feats["mean_pass_len"] = np.mean(lengths)
    feats["forward_ratio"] = np.mean(fwd)
    feats["mean_progression"] = np.mean([ (r[6]-r[4])*flip for r in comp ])
    feats["mean_start_x"] = np.mean(sx)  # buildup origin (lower => from back)
    feats["mean_end_x"] = np.mean(ex)
    feats["side_bias"] = (np.mean(sy) - 40.0) / 40.0  # -1 left .. +1 right
    feats["lateral_ratio"] = np.mean([abs(r[7]-r[5]) / max(1, math.hypot(r[6]-r[4], r[7]-r[5])) for r in comp])
    # position-group involvement (possession style: defenders/GK passing)
    grp = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for r in comp:
        g = POS_GROUPS.get(pos_of.get(r[0], ""), "MID")
        grp[g] += 1
    tot = max(1, sum(grp.values()))
    feats["pct_pass_by_DEF"] = grp["DEF"] / tot
    feats["pct_pass_by_GK"] = grp["GK"] / tot
    feats["pct_pass_by_MID"] = grp["MID"] / tot
    feats["pct_pass_by_FWD"] = grp["FWD"] / tot
    # GCN embedding (structural fingerprint)
    degfeat = np.column_stack([indeg, outdeg, (indeg+outdeg)])
    degfeat = degfeat / (degfeat.max(0) + 1e-9)
    emb = gcn_embedding(W, degfeat, seed=COMP+team_id)
    return feats, emb, (G, pid_list, pos_of, A, idx)

rows, embs, graphs = [], {}, {}
for tid in team_passes:
    out = build_features(tid)
    if out is None:
        continue
    feats, emb, graph = out
    feats["team_id"] = tid
    feats["team"] = team_name.get(tid, str(tid))
    rows.append(feats)
    embs[tid] = emb
    graphs[tid] = graph

df = pd.DataFrame(rows)
print("teams with features:", len(df))
feat_cols = [c for c in df.columns if c not in ("team_id", "team")]
X = df[feat_cols].astype(float).values
Xs = StandardScaler().fit_transform(X)
Emb = np.vstack([embs[t] for t in df["team_id"]])
EmbS = StandardScaler().fit_transform(Emb)
# Representation = PCA of interpretable handcrafted features
# (the GCN embedding is computed as a structural fingerprint and reported,
#  but the primary clustering representation uses the interpretable features).
pca = PCA(n_components=min(12, Xs.shape[1]), random_state=42)
Xpca = pca.fit_transform(Xs)
Xcombo = Xpca
print("clustering representation dim (PCA):", Xcombo.shape[1],
      "| explained var:", round(float(pca.explained_variance_ratio_.sum()), 3))

# ---------- clustering: choose k in a sensible [4,6] range by silhouette ----------
best_s, bestk = -1, 5
for k in [4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Xcombo)
    s = silhouette_score(Xcombo, km.labels_)
    frac = pd.Series(km.labels_).value_counts(normalize=True).max()
    print(f"  k={k} silhouette={s:.3f} max_cluster_frac={frac:.2f}")
    if s > best_s:
        best_s, bestk = s, k
print("chosen k =", bestk, "silhouette =", round(best_s, 3))
km = KMeans(n_clusters=bestk, random_state=42, n_init=10).fit(Xcombo)
df["cluster"] = km.labels_

# ---------- interpretability (on interpretable handcrafted features) ----------
clf = RandomForestClassifier(n_estimators=400, random_state=42)
clf.fit(Xs, df["cluster"])
imp = clf.feature_importances_
feat_names = list(feat_cols)
imp_df = pd.DataFrame({"feature": feat_names, "importance": imp}).sort_values("importance", ascending=False)

# cluster centroid profiles (z-scored within features)
Xz = pd.DataFrame(Xs, columns=feat_cols)
prof = Xz.groupby(df["cluster"]).mean()
print("\ncluster sizes:")
print(df.groupby("cluster")["team"].apply(list).to_dict())

# name prototypes from dominant features
def proto_name(row):
    tags = []
    if row.get("completion_rate", 0) > prof["completion_rate"].mean() and row.get("mean_progression", 0) > prof["mean_progression"].mean():
        tags.append("possession-buildup")
    if row.get("forward_ratio", 0) > prof["forward_ratio"].mean() and row.get("mean_pass_len", 0) > prof["mean_pass_len"].mean():
        tags.append("direct-counter")
    if row.get("pct_pass_by_DEF", 0) > prof["pct_pass_by_DEF"].mean():
        tags.append("defender-involved")
    if row.get("density_dir", 0) > prof["density_dir"].mean():
        tags.append("high-connectivity")
    if row.get("modularity", 0) > prof["modularity"].mean():
        tags.append("modular")
    return "-".join(tags[:2]) if tags else "balanced"
proto_labels = {}
for c in sorted(df["cluster"].unique()):
    sub = prof.loc[c]
    proto_labels[c] = proto_name(sub)
df["proto"] = df["cluster"].map(proto_labels)
print("\nprototype labels:", proto_labels)

# ---------- save results ----------
df.to_csv(os.path.join(RES, "team_features.csv"), index=False)
imp_df.to_csv(os.path.join(RES, "feature_importance.csv"), index=False)
prof.to_csv(os.path.join(RES, "cluster_profiles.csv"))
out = {
    "n_teams": int(len(df)),
    "n_features": int(Xcombo.shape[1]),
    "best_k": int(bestk),
    "silhouette": float(best_s),
    "prototype_labels": {str(k): v for k, v in proto_labels.items()},
    "clusters": {str(c): df[df["cluster"] == c]["team"].tolist() for c in sorted(df["cluster"].unique())},
    "top_features": imp_df.head(12).to_dict("records"),
}
json.dump(out, open(os.path.join(RES, "clusters.json"), "w"), ensure_ascii=False, indent=2)

# ---------- downstream demo: predict match result from team embeddings ----------
Xtr = StandardScaler().fit_transform(Xcombo)
team_to_idx = {tid: i for i, tid in enumerate(df["team_id"])}
Xdemo, ydemo = [], []
for mid, (h, a, lab) in match_results.items():
    if h in team_to_idx and a in team_to_idx:
        Xdemo.append(np.hstack([Xtr[team_to_idx[h]], Xtr[team_to_idx[a]]]))
        ydemo.append(lab)
Xdemo, ydemo = np.array(Xdemo), np.array(ydemo)
if len(Xdemo) > 10:
    m = RandomForestClassifier(n_estimators=300, random_state=42)
    cv = cross_val_score(m, Xdemo, ydemo, cv=5)
    print(f"\n[downstream] match-result prediction CV acc = {cv.mean():.3f} +/- {cv.std():.3f} (n={len(Xdemo)})")
    out["downstream_cv_acc"] = float(cv.mean())
    out["downstream_n"] = int(len(Xdemo))
    json.dump(out, open(os.path.join(RES, "clusters.json"), "w"), ensure_ascii=False, indent=2)

# ================= VISUALIZATIONS =================
# 1) PCA scatter of clusters
pca = PCA(n_components=2).fit_transform(Xcombo)
plt.figure(figsize=(8, 6))
cmap = plt.cm.tab10
for c in sorted(df["cluster"].unique()):
    msk = df["cluster"] == c
    plt.scatter(pca[msk.values, 0], pca[msk.values, 1],
                label=f"C{c}: {proto_labels[c]}", s=70, alpha=0.8, color=cmap(c % 10))
    for i in np.where(msk.values)[0]:
        plt.annotate(df["team"].iloc[i], (pca[i, 0], pca[i, 1]), fontsize=7, alpha=0.7)
plt.title("Tactical prototypes (PCA of team embeddings)")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "pca_scatter.png"), dpi=130)
plt.close()

# 2) centroid heatmap (top features)
topf = imp_df["feature"].head(12).tolist()
# map gcn names back to feat_cols for profiling
prof_plot = prof[[c for c in topf if c in prof.columns]]
if prof_plot.shape[1] == 0:
    prof_plot = prof.iloc[:, :12]
plt.figure(figsize=(10, 5))
im = plt.imshow(prof_plot.values, aspect="auto", cmap="coolwarm", vmin=-1.5, vmax=1.5)
plt.colorbar(im, label="z-score")
plt.yticks(range(prof_plot.shape[0]), [f"C{c}: {proto_labels[c]}" for c in prof_plot.index])
plt.xticks(range(prof_plot.shape[1]), prof_plot.columns, rotation=90, fontsize=7)
plt.title("Cluster centroid feature profiles (interpretability)")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "centroid_heatmap.png"), dpi=130)
plt.close()

# 3) feature importance bars
plt.figure(figsize=(8, 5))
top = imp_df.head(15)
plt.barh(top["feature"][::-1], top["importance"][::-1])
plt.title("Top features driving tactical prototypes (RF importance)")
plt.xlabel("importance")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "importance.png"), dpi=130)
plt.close()

# 4) passing networks for representative teams (closest to each centroid)
def plot_network(tid, fname):
    G, pid_list, pos_of, A, idx = graphs[tid]
    n = len(pid_list)
    # pitch coordinates for nodes: use mean pass location per player (approx via recipient/passer)
    # simpler: place by position groups on a 105x68 pitch
    pos_cat = {p: POS_GROUPS.get(pos_of.get(p, ""), "MID") for p in pid_list}
    xpos = {"GK": 6, "DEF": 28, "MID": 55, "FWD": 82}
    ygrid = {"GK": [34], "DEF": [14, 27, 41, 54], "MID": [14, 27, 41, 54], "FWD": [20, 48]}
    used = {k: 0 for k in ygrid}
    coords = {}
    for p in pid_list:
        cat = pos_cat[p]
        i = used[cat] % len(ygrid[cat])
        coords[p] = (xpos[cat], ygrid[cat][i])
        used[cat] += 1
    fig, ax = plt.subplots(figsize=(7, 5))
    # pitch
    ax.set_xlim(0, 105); ax.set_ylim(0, 68)
    ax.add_patch(plt.Rectangle((0, 0), 105, 68, fill=False, color="gray"))
    ax.add_patch(plt.Rectangle((0, 24), 16.5, 20, fill=False, color="gray"))
    ax.add_patch(plt.Rectangle((88.5, 24), 16.5, 20, fill=False, color="gray"))
    ax.plot(52.5, 0, 52.5, 68, color="gray", lw=1)
    # edges
    for i in range(n):
        for j in range(n):
            w = A[i, j]
            if w > 0:
                pi, pj = pid_list[i], pid_list[j]
                xs, ys = coords[pi]; xe, ye = coords[pj]
                ax.plot([xs, xe], [ys, ye], color="blue", alpha=min(0.8, 0.1 + w / A.max() * 0.7), lw=min(4, 0.5 + w / A.max() * 3))
    # nodes
    deg = A.sum(1) + A.sum(0)
    for p in pid_list:
        x, y = coords[p]
        sz = 120 + 600 * deg[idx[p]] / deg.max()
        ax.scatter(x, y, s=sz, color="red", edgecolor="black", zorder=3)
    ax.set_title(f"Passing network: {team_name.get(tid, tid)}\n(cluster {df[df.team_id==tid]['cluster'].values[0]} - {df[df.team_id==tid]['proto'].values[0]})")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close()

rep = []
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    # closest to centroid in combined space
    cen = Xcombo[df["cluster"] == c].mean(0)
    d = np.linalg.norm(Xcombo[df["cluster"] == c] - cen, axis=1)
    tid = sub.iloc[np.argmin(d)]["team_id"]
    rep.append(tid)
for tid in rep[:6]:
    plot_network(tid, f"network_{team_name.get(tid, tid).replace(' ','_')}.png")

print("\nDONE. Figures in", FIG, "| Results in", RES)
print("Representative teams plotted:", [team_name.get(t, t) for t in rep[:6]])
