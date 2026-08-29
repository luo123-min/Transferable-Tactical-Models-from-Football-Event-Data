#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validate_360.py — three-sixty 特征验证与可解释分析
==================================================
1. 新颖性验证：将 360 特征与 pipeline_v2 的事件流统一特征做相关分析，
   低相关即证明 360 提供了事件流之外的"局部防守形态/逼抢"独立信号。
2. 聚类：对 360 特征（持球方 + 防守方视角）做 PCA + KMeans，
   归纳"防守/逼抢风格"原型，并用质心剖面解释。
3. 可视化：
   - freeze_frame 单帧示意图（传球线路占据 + 局部防守结构）
   - 360 原型 PCA 散点
   - 原型质心热图
   - 360 vs 事件流 相关性热图（新颖性证据）
"""
import os, json, zipfile
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

HERE = os.path.dirname(__file__)
ZIP = os.path.join(HERE, "open-data-master.zip")
R360 = os.path.join(HERE, "results_360")
RFULL = os.path.join(HERE, "results_full")
FIG = os.path.join(HERE, "figures_360")
os.makedirs(FIG, exist_ok=True)

POSS_COLS = ["poss_press_min_opp_dist", "poss_press_mean_opp_dist",
             "poss_support_mean_dist", "poss_support_n_close10",
             "poss_lane_min_opp_dist", "poss_lane_blocked_n", "poss_pass_length"]
DEF_COLS = ["def_compact_area", "def_width", "def_depth",
            "def_mean_dist_to_ball", "def_nearest_to_ball", "def_n_players"]

# ---------------------------------------------------------------------------
# 1) 加载数据
# ---------------------------------------------------------------------------
def load():
    f360 = os.path.join(R360, "team_features_360.csv")
    df360 = pd.read_csv(f360)
    print(f"[360-val] loaded {len(df360)} team-seasons from {f360}")
    funi = os.path.join(RFULL, "unified_team_features.csv")
    dfu = pd.read_csv(funi) if os.path.exists(funi) else None
    if dfu is not None:
        print(f"[360-val] loaded {len(dfu)} team-seasons from unified table")
    return df360, dfu

# ---------------------------------------------------------------------------
# 2) 新颖性：360 特征 vs 事件流特征 相关矩阵
# ---------------------------------------------------------------------------
def novelty_correlation(df360, dfu):
    if dfu is None:
        print("[360-val] no unified table -> skip novelty correlation")
        return None
    key = ["team_id", "competition_id", "season_id"]
    m = df360.merge(dfu, on=key, how="inner", suffixes=("_360", "_ev"))
    print(f"[360-val] merged for novelty: {len(m)} team-seasons")
    # 事件流侧选代表特征
    ev_cols = [c for c in dfu.columns if c in
               ("passes_per_match", "mean_pass_length", "short_pass_frac",
                "long_pass_frac", "pressure_per_match", "interception_per_match",
                "block_per_match", "clearance_per_match", "shot_per_match",
                "carry_per_match", "ball_recovery_per_match", "density",
                "transitivity", "mean_degree", "mean_betweenness", "mean_closeness")]
    feats360 = POSS_COLS + DEF_COLS
    sub = m[feats360 + ev_cols].dropna()
    corr = sub[feats360].corrwith(sub[ev_cols], axis=0) if False else None
    # 两两相关矩阵（只取每侧代表列，避免图过大）
    C = sub[feats360 + ev_cols].corr()
    # 保存
    C.to_csv(os.path.join(R360, "corr_360_vs_eventstream.csv"))
    print(f"[360-val] correlation matrix saved ({C.shape})")
    # 平均绝对相关（360 各特征 vs 全部事件流特征）——越低越"新颖"
    abs_c = C.loc[feats360, ev_cols].abs().mean(axis=1)
    print("\n[360-val] 360 特征与事件流特征的平均|相关|（越低=越独立的新信号）:")
    for k, v in abs_c.sort_values().items():
        print(f"   {k:32s} {v:.3f}")
    return C, feats360, ev_cols, m

def plot_correlation(C, feats360, ev_cols):
    plt.figure(figsize=(10, 8))
    mat = C.loc[feats360, ev_cols].values
    im = plt.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.xticks(range(len(ev_cols)), ev_cols, rotation=90, fontsize=7)
    plt.yticks(range(len(feats360)), feats360, fontsize=7)
    plt.title("360 features vs Event-stream features\n(low correlation = novel spatial signal)", fontsize=10)
    plt.tight_layout()
    p = os.path.join(FIG, "corr_360_vs_eventstream.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"[360-val] saved {p}")
    return p

# ---------------------------------------------------------------------------
# 3) 聚类：360 防守/逼抢风格原型
# ---------------------------------------------------------------------------
def cluster(df360):
    feats = POSS_COLS + DEF_COLS
    X = df360[feats].astype(float).fillna(df360[feats].mean()).values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(10, len(feats))).fit(Xs)
    evr = np.cumsum(pca.explained_variance_ratio_)
    kpca = int(np.searchsorted(evr, 0.90) + 1)
    kpca = max(2, min(kpca, len(feats)))
    print(f"[360-val] PCA: {kpca} comps explain >=90% variance "
          f"(1st={pca.explained_variance_ratio_[0]:.2f}, 2nd={pca.explained_variance_ratio_[1]:.2f})")
    Z = pca.transform(Xs)[:, :kpca]
    best_k, best_s = 3, -1
    for k in range(3, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(Z)
        s = silhouette_score(Z, km.labels_)
        print(f"   k={k} silhouette={s:.3f}")
        if s > best_s:
            best_k, best_s = k, s
    print(f"[360-val] chosen k={best_k} silhouette={best_s:.3f}")
    km = KMeans(n_clusters=best_k, random_state=42, n_init=10).fit(Z)
    df360 = df360.copy()
    df360["cluster"] = km.labels_
    # 质心剖面（z-score）
    prof = df360.groupby("cluster")[feats].mean()
    prof_z = (prof - prof.mean()) / prof.std()
    # 命名
    names = prototype_names(prof)
    print("\n[360-val] 360 战术/防守原型:")
    for c in sorted(prof.index):
        sub = df360[df360.cluster == c]
        print(f"  C{c} [{names[c]}] n={len(sub)}  e.g.: "
              + ", ".join(sub.sort_values('def_nearest_to_ball').head(4)['team_name'].tolist()))
    return df360, prof, prof_z, names, Z, pca, feats

def prototype_names(prof):
    names = {}
    press_mean = prof["def_nearest_to_ball"].mean() + prof["poss_press_min_opp_dist"].mean()
    compact_mean = prof["def_compact_area"].mean()
    for c in prof.index:
        r = prof.loc[c]
        press = r["def_nearest_to_ball"] + r["poss_press_min_opp_dist"]
        compact = r["def_compact_area"]
        if press < press_mean:
            base = "High-press"
        else:
            base = "Low-block"
        base += " Compact" if compact > compact_mean else " Loose"
        names[c] = base
    return names

def plot_clusters(Z, labels, names):
    plt.figure(figsize=(8, 6))
    uniq = sorted(np.unique(labels))
    cmap = plt.cm.tab10
    for i, c in enumerate(uniq):
        m = labels == c
        plt.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.7,
                    color=cmap(i % 10), label=f"C{c} {names[c]}")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("360 spatial-style prototypes (PCA of freeze-frame features)")
    plt.legend(fontsize=7, loc="best")
    plt.tight_layout()
    p = os.path.join(FIG, "pca_scatter_360.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"[360-val] saved {p}")
    return p

def plot_profiles(prof_z, names):
    plt.figure(figsize=(11, 5))
    im = plt.imshow(prof_z.values, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    plt.colorbar(im, fraction=0.03, pad=0.02)
    plt.yticks(range(len(prof_z)), [f"C{c} {names[c]}" for c in prof_z.index],
               fontsize=8)
    plt.xticks(range(len(prof_z.columns)), prof_z.columns, rotation=90, fontsize=7)
    plt.title("360 prototype centroid profiles (z-scored)")
    plt.tight_layout()
    p = os.path.join(FIG, "centroid_heatmap_360.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"[360-val] saved {p}")
    return p

# ---------------------------------------------------------------------------
# 4) 单帧冻结帧示意图（论文好图）
# ---------------------------------------------------------------------------
def find_interesting_pass(z, max_files=40):
    """找一个传球线路被对手贴近（lane_min 小）且球员数充足的冻结帧"""
    names = [n for n in z.namelist()
             if n.startswith("open-data-master/data/three-sixty/") and n.endswith(".json")]
    best = None
    for n in names[:max_files]:
        mid = n.split("/")[-1].replace(".json", "")
        try:
            t3 = json.loads(z.read(n))
            evts = json.loads(z.read(f"open-data-master/data/events/{mid}.json"))
        except KeyError:
            continue
        ev_by_id = {e["id"]: e for e in evts}
        for e360 in t3:
            ev = ev_by_id.get(e360.get("event_uuid"))
            if not ev or ev.get("type", {}).get("name") != "Pass":
                continue
            end = ev.get("pass", {}).get("end_location")
            if not end:
                continue
            ff = e360.get("freeze_frame") or []
            if len(ff) < 8:
                continue
            act = [p for p in ff if p.get("actor")]
            if not act:
                continue
            focal = act[0]["location"]
            opp = [p["location"] for p in ff if not p.get("teammate")]
            # lane min dist
            def pdist(p, a, b):
                ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
                dx, dy = bx-ax, by-ay
                if dx == 0 and dy == 0:
                    return ((px-ax)**2+(py-ay)**2)**0.5
                t = max(0, min(1, ((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy)))
                return ((px-(ax+t*dx))**2+(py-(ay+t*dy))**2)**0.5
            seg = [pdist(o, focal, [float(end[0]), float(end[1])]) for o in opp]
            if seg and min(seg) < 1.2:   # 线路被贴身封堵
                score = min(seg)
                if best is None or score < best[0]:
                    best = (score, mid, e360, ev, focal, [float(end[0]), float(end[1])])
    return best

def plot_freeze_frame(best):
    score, mid, e360, ev, focal, end = best
    ff = e360["freeze_frame"]
    mates = [p for p in ff if p.get("teammate")]
    opps = [p for p in ff if not p.get("teammate")]
    plt.figure(figsize=(9, 6))
    # 球场
    plt.gca().add_patch(plt.Rectangle((0, 0), 120, 80, fill=False, ec="grey", lw=1.5))
    plt.plot([60, 60], [0, 80], c="grey", lw=1, ls="--")
    # 球员
    for p in mates:
        plt.scatter(p["location"][0], p["location"][1], c="#1f77b4", s=60,
                    edgecolor="k", zorder=3, label="teammate" if p is mates[0] else "")
    for p in opps:
        plt.scatter(p["location"][0], p["location"][1], c="#d62728", s=60,
                    edgecolor="k", zorder=3, label="opponent" if p is opps[0] else "")
    # 传球线路
    plt.plot([focal[0], end[0]], [focal[1], end[1]], c="#2ca02c", lw=2.5, ls="--",
             zorder=2, label="pass")
    # 最近封堵者到线路
    opp_loc = [p["location"] for p in opps]
    def pdist(p, a, b):
        ax, ay, bx, by, px, py = a[0], a[1], b[0], b[1], p[0], p[1]
        dx, dy = bx-ax, by-ay
        if dx == 0 and dy == 0:
            return ((px-ax)**2+(py-ay)**2)**0.5, (px, py)
        t = max(0, min(1, ((px-ax)*dx+(py-ay)*dy)/(dx*dx+dy*dy)))
        cx, cy = ax+t*dx, ay+t*dy
        return ((px-cx)**2+(py-cy)**2)**0.5, (cx, cy)
    seg = [pdist(o, focal, end) for o in opp_loc]
    bi = int(np.argmin([s[0] for s in seg]))
    md, (cx, cy) = seg[bi]
    plt.scatter([cx], [cy], c="#d62728", marker="x", s=120, lw=2.5, zorder=4)
    plt.plot([opp_loc[bi][0], cx], [opp_loc[bi][1], cy], c="#d62728", lw=1.2, ls=":")
    plt.annotate(f"lane gap={md:.1f}m", (cx, cy), textcoords="offset points",
                 xytext=(6, 6), fontsize=8, color="#d62728")
    # actor
    plt.scatter([focal[0]], [focal[1]], c="#ff7f0e", s=120, marker="*",
                edgecolor="k", zorder=5, label="ball carrier")
    # compactness of opponents
    ox = [p[0] for p in opp_loc]; oy = [p[1] for p in opp_loc]
    plt.title(f"360 freeze-frame (match {mid}): passing-lane occupation\n"
              f"opp compact area~{np.ptp(ox)*np.ptp(oy):.0f}  opp width~{np.ptp(oy):.0f}m", fontsize=10)
    plt.xlabel("x (0-120, possessing team -> +x)"); plt.ylabel("y (0-80)")
    plt.xlim(-3, 123); plt.ylim(-3, 83)
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    p = os.path.join(FIG, "freeze_frame_example.png")
    plt.savefig(p, dpi=140); plt.close()
    print(f"[360-val] saved freeze-frame {p} (lane gap={md:.2f}m)")
    return p

# ---------------------------------------------------------------------------
def compare_eventstream_clusters(m, labels360):
    """事件流特征单独聚类，与 360 聚类比较吻合度（ARI）。
    低 ARI => 360 揭示了事件流之外的互补结构。"""
    from sklearn.metrics import adjusted_rand_score
    from sklearn.cluster import KMeans
    ev_cols = [c for c in m.columns if c in
               ("passes_per_match", "mean_pass_length", "short_pass_frac",
                "long_pass_frac", "pressure_per_match", "interception_per_match",
                "block_per_match", "clearance_per_match", "shot_per_match",
                "carry_per_match", "ball_recovery_per_match", "density",
                "transitivity", "mean_degree", "mean_betweenness", "mean_closeness")]
    Xev = m[ev_cols].astype(float).fillna(m[ev_cols].mean()).values
    Xev = StandardScaler().fit_transform(Xev)
    Zev = PCA(n_components=0.9).fit_transform(Xev)
    kev = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(Zev)
    ari = adjusted_rand_score(labels360, kev)
    print(f"\n[360-val] ARI(360 clusters, event-stream clusters) = {ari:.3f} "
          f"(0=随机吻合, 1=完全一致；越低说明 360 揭示互补结构)")
    return ari

def main():
    df360, dfu = load()
    res = novelty_correlation(df360, dfu)
    if res is not None:
        Cmat, f360, evc, m = res
        plot_correlation(Cmat, f360, evc)
    df360, prof, prof_z, names, Z, pca, feats = cluster(df360)
    plot_clusters(Z, df360["cluster"].values, names)
    plot_profiles(prof_z, names)
    if dfu is not None:
        ari = compare_eventstream_clusters(m, df360["cluster"].values)
    # 冻结帧图
    z = zipfile.ZipFile(ZIP)
    best = find_interesting_pass(z)
    if best:
        plot_freeze_frame(best)
    # 保存聚类结果
    df360.to_csv(os.path.join(R360, "team_features_360_clustered.csv"), index=False)
    prof.to_csv(os.path.join(R360, "cluster_profiles_360.csv"))
    print("[360-val] done.")

if __name__ == "__main__":
    main()
