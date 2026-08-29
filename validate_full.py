"""
validate_full.py — sanity check on the unified feature table.

Loads results_full/unified_team_features.csv, runs PCA + KMeans across the
cross-league corpus, and reports whether coherent tactical prototypes emerge
at scale (vs the earlier 35-match World-Cup-only prototype).

Outputs:
  results_full/pca_scatter_full.png
  results_full/cluster_profiles_full.csv
  prints cluster sizes, top driving features, example teams.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

CSV = "results_full/unified_team_features.csv"
META = ["team_id", "team_name", "competition_id", "season_id", "competition_name",
        "season_name", "country_name", "competition_gender", "n_matches"]

df = pd.read_csv(CSV)

# Balance the corpus so no single league dominates the prototypes.
# (In this slice La Liga has far more team-seasons than other leagues.)
cap = 45
parts = []
for name, g in df.groupby("competition_name"):
    parts.append(g.sample(min(len(g), cap), random_state=42))
balanced = pd.concat(parts, ignore_index=True)
print(f"raw team-seasons={len(df)} -> balanced={len(balanced)} "
      f"(cap {cap}/league: " +
      ", ".join(f"{k} {v}" for k, v in balanced['competition_name'].value_counts().items()) + ")")
df = balanced.reset_index(drop=True)

feat = [c for c in df.columns if c not in META]
X = df[feat].astype(float).fillna(0).values
Xs = StandardScaler().fit_transform(X)

pca = PCA(n_components=0.95, random_state=42)
Z = pca.fit_transform(Xs)
print(f"PCA: {Z.shape[1]} components explain 95% variance")

best_k, best_s, best_labels = 3, -1, None
for k in range(3, 9):
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
    s = silhouette_score(Z, km.labels_)
    print(f"  k={k} silhouette={s:.3f}")
    if s > best_s:
        best_k, best_s, best_labels = k, s, km.labels_
print(f"chosen k={best_k} silhouette={best_s:.3f}")
df["cluster"] = best_labels

# cluster profiles: mean z-scored feature per cluster
prof = pd.DataFrame(Xs, columns=feat)
prof["cluster"] = best_labels
cent = prof.groupby("cluster").mean()

# name prototypes by top positive/negative drivers
def name_proto(row):
    top = row.abs().sort_values(ascending=False).head(3).index.tolist()
    return "+".join(top)
labels = {c: name_proto(cent.loc[c]) for c in cent.index}

print("\n=== CLUSTER SIZES & EXAMPLE TEAMS ===")
for c in sorted(df["cluster"].unique()):
    sub = df[df["cluster"] == c]
    ex = ", ".join(sub["team_name"].value_counts().index[:6])
    print(f"  cluster {c} (n={len(sub)}): {labels[c]}")
    print(f"      e.g. {ex}")

# top driving features overall (variance across clusters)
drive = cent.abs().max(axis=0).sort_values(ascending=False).head(10)
print("\n=== TOP DISCRIMINATIVE FEATURES (cluster spread) ===")
for f, v in drive.items():
    print(f"  {f}: {v:.2f}")

# save profiles
cent.to_csv("results_full/cluster_profiles_full.csv")
print("\nsaved cluster_profiles_full.csv")

# figure: PCA scatter colored by cluster, labeled by competition
plt.figure(figsize=(8, 6))
cmap = plt.cm.tab10
for c in sorted(df["cluster"].unique()):
    s = df[df["cluster"] == c]
    plt.scatter(Z[s.index, 0], Z[s.index, 1], s=18, alpha=0.7,
                color=cmap(c % 10), label=f"C{c}")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.title(f"Cross-league tactical prototypes (k={best_k}, sil={best_s:.2f})")
plt.legend(fontsize=8, markerscale=1.5)
plt.tight_layout()
plt.savefig("results_full/pca_scatter_full.png", dpi=130)
print("saved pca_scatter_full.png")

# also: how many clusters contain multiple leagues (cross-league transfer evidence)
print("\n=== CROSS-LEAGUE TRANSFER (clusters spanning >1 competition) ===")
for c in sorted(df["cluster"].unique()):
    comps = df[df["cluster"] == c]["competition_name"].value_counts()
    if len(comps) > 1:
        print(f"  C{c}: " + ", ".join(f"{k}({v})" for k, v in comps.items()))
