#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pipeline_360.py — StatsBomb 360 (three-sixty) 冻结帧战术特征管线
================================================================
从 open-data-master.zip 直接读取 three-sixty 冻结帧，结合 events 文件，
为每支球队在「球附近的局部防守结构 / 逼抢强度 / 传球线路占据」维度构建特征，
按 球队 × 赛事季 聚合，输出与 pipeline_v2.py 同键 (team_id, competition_id, season_id)
的统一特征表，便于合并分析。

核心思路：
  一次 360 冻结帧 = 某事件发生时、visible_area 内所有球员站位 (120x80 坐标系，
  持球方恒朝 +x)。一帧同时贡献两条记录：
    - 持球方 (event team)  -> 进攻支持 / 被逼抢强度 / 传球线路占据
    - 防守方 (另一队)      -> 局部防守紧凑度 / 防线宽度深度 / 逼抢侵略性
  两条记录按 team_id 分别聚合到 team-season。

用法：
  python pipeline_360.py                 # 处理全部 426 场 360 比赛
  python pipeline_360.py --limit 50      # 仅前 50 场（调试）
"""
import os, json, argparse, collections
import numpy as np
import pandas as pd
import zipfile
from math import sqrt

ZIP = os.path.join(os.path.dirname(__file__), "open-data-master.zip")
OUTDIR = os.path.join(os.path.dirname(__file__), "results_360")
MATCH_INDEX_CACHE = os.path.join(OUTDIR, "match_index.json")
os.makedirs(OUTDIR, exist_ok=True)

# ----------------------------------------------------------------------------
# 几何工具
# ----------------------------------------------------------------------------
def _dist(a, b):
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def point_to_segment_dist(p, a, b):
    """点 p 到线段 ab 的最短距离"""
    ax, ay = a; bx, by = b; px, py = p
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return _dist(p, a)
    t = ((px - ax) * dx + (py - ay) * dy) / float(dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    cx, cy = ax + t * dx, ay + t * dy
    return sqrt((px - cx) ** 2 + (py - cy) ** 2)

def convex_hull_area(pts):
    """点集凸包面积 (shoelace)。<3 点返回包围盒面积兜底。"""
    pts = np.array(pts, dtype=float)
    if len(pts) < 3:
        if len(pts) == 0:
            return 0.0
        if len(pts) == 1:
            return 0.0
        # 2 点 -> 退化，用包围盒
        return 0.0
    # Andrew monotone chain
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    n = len(hull)
    for i in range(n):
        x1, y1 = hull[i]
        x2, y2 = hull[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

# ----------------------------------------------------------------------------
# match 索引：match_id -> (cid, sid, home_id, home_name, away_id, away_name)
# ----------------------------------------------------------------------------
def build_match_index(z):
    if os.path.exists(MATCH_INDEX_CACHE):
        return json.load(open(MATCH_INDEX_CACHE))
    idx = {}
    for n in z.namelist():
        if not n.startswith("open-data-master/data/matches/") or not n.endswith(".json"):
            continue
        parts = n.split("/")
        cid, sid = int(parts[3]), int(parts[4].replace(".json", ""))
        for m in json.loads(z.read(n)):
            h = m.get("home_team", {})
            a = m.get("away_team", {})
            idx[str(m["match_id"])] = {
                "cid": cid, "sid": sid,
                "home_id": h.get("home_team_id"), "home_name": h.get("home_team_name"),
                "away_id": a.get("away_team_id"), "away_name": a.get("away_team_name"),
            }
    json.dump(idx, open(MATCH_INDEX_CACHE, "w"), ensure_ascii=False)
    print(f"[match_index] built: {len(idx)} matches -> {MATCH_INDEX_CACHE}")
    return idx

# ----------------------------------------------------------------------------
# 单帧特征（按角色只算相关特征，语义清晰）
# ----------------------------------------------------------------------------
def frame_features_poss(ff, focal, end):
    """持球方（actor 所在队）视角特征"""
    opponents = [p["location"] for p in ff if not p.get("teammate")]
    teammates = [p["location"] for p in ff if p.get("teammate") and not p.get("actor")]
    feats = {}
    if opponents:
        opp_d = [_dist(focal, o) for o in opponents]
        feats["press_min_opp_dist"] = float(min(opp_d))     # 最近对手距球（被逼抢紧度）
        feats["press_mean_opp_dist"] = float(np.mean(opp_d))
    else:
        feats["press_min_opp_dist"] = np.nan
        feats["press_mean_opp_dist"] = np.nan
    if teammates:
        tm_d = [_dist(focal, t) for t in teammates]
        feats["support_mean_dist"] = float(np.mean(tm_d))   # 队友平均支持距离
        feats["support_n_close10"] = float(sum(1 for d in tm_d if d < 10.0))
    else:
        feats["support_mean_dist"] = np.nan
        feats["support_n_close10"] = 0.0
    if end is not None:
        seg = [point_to_segment_dist(o, focal, end) for o in opponents] if opponents else []
        feats["lane_min_opp_dist"] = float(min(seg)) if seg else np.nan   # 传球线路被占据程度
        feats["lane_blocked_n"] = float(sum(1 for d in seg if d < 2.0))   # 线路 2m 内对手数
        feats["pass_length"] = float(_dist(focal, end))
    else:
        feats["lane_min_opp_dist"] = np.nan
        feats["lane_blocked_n"] = np.nan
        feats["pass_length"] = np.nan
    feats["n_players_visible"] = float(len(ff))
    return feats

def frame_features_def(ff, focal):
    """防守方（actor 的对手，即本帧 teammate=False 的球员）视角特征"""
    our = [p["location"] for p in ff if not p.get("teammate")]
    feats = {}
    if not our:
        feats.update({"compact_area": np.nan, "width": np.nan, "depth": np.nan,
                      "mean_dist_to_ball": np.nan, "nearest_to_ball": np.nan,
                      "n_players": np.nan})
        feats["n_players_visible"] = float(len(ff))
        return feats
    ox = [p[0] for p in our]; oy = [p[1] for p in our]
    feats["width"] = float(max(oy) - min(oy)) if len(oy) > 1 else 0.0   # y 跨度（防线宽度）
    feats["depth"] = float(max(ox) - min(ox)) if len(ox) > 1 else 0.0   # x 跨度（防线纵深）
    feats["compact_area"] = float(convex_hull_area(our))                # 局部紧凑度
    d = [_dist(focal, o) for o in our]
    feats["mean_dist_to_ball"] = float(np.mean(d))   # 防守方平均距球（逼抢侵略性）
    feats["nearest_to_ball"] = float(min(d))         # 最近防守者距球
    feats["n_players"] = float(len(our))
    feats["n_players_visible"] = float(len(ff))
    return feats

# ----------------------------------------------------------------------------
# 处理单场比赛
# ----------------------------------------------------------------------------
def process_match(args):
    mid, zip_path = args
    z = zipfile.ZipFile(zip_path)
    ev_path = f"open-data-master/data/events/{mid}.json"
    t3_path = f"open-data-master/data/three-sixty/{mid}.json"
    try:
        evts = json.loads(z.read(ev_path))
    except (KeyError, json.JSONDecodeError):
        return []
    try:
        t3 = json.loads(z.read(t3_path))
    except (KeyError, json.JSONDecodeError):
        return []
    ev_by_id = {e["id"]: e for e in evts}
    match_index = json.load(open(MATCH_INDEX_CACHE))
    mi = match_index.get(str(mid))
    if mi is None:
        return []
    cid, sid = mi["cid"], mi["sid"]
    teams = {mi["home_id"]: mi["home_name"], mi["away_id"]: mi["away_name"]}
    event_team_ids = set(teams.keys())

    records = []
    for e360 in t3:
        eid = e360.get("event_uuid")
        ev = ev_by_id.get(eid)
        if ev is None:
            continue
        ff = e360.get("freeze_frame") or []
        if not ff:
            continue
        eteam = ev.get("team", {}).get("id")
        if eteam not in event_team_ids:
            continue
        # focal = actor 位置（无则取 event location）
        actor_pts = [p["location"] for p in ff if p.get("actor")]
        focal = actor_pts[0] if actor_pts else ev.get("location")
        if not focal:
            continue
        # 传球终点
        end = None
        if ev.get("type", {}).get("name") == "Pass":
            pl = ev.get("pass", {}).get("end_location")
            if pl:
                end = [float(pl[0]), float(pl[1])]
        # 防守方 = 另一队
        def_team = [t for t in event_team_ids if t != eteam][0]
        # 持球方记录（进攻/被逼抢视角）
        r_poss = {"team_id": eteam, "team_name": teams.get(eteam), "cid": cid, "sid": sid,
                  "role": "poss"}
        r_poss.update(frame_features_poss(ff, focal, end))
        records.append(r_poss)
        # 防守方记录（局部防守结构视角）
        r_def = {"team_id": def_team, "team_name": teams.get(def_team), "cid": cid, "sid": sid,
                 "role": "def"}
        r_def.update(frame_features_def(ff, focal))
        records.append(r_def)
    return records

# ----------------------------------------------------------------------------
# 聚合
# ----------------------------------------------------------------------------
def aggregate(records):
    # 按 (team_id, cid, sid, role) 求均值
    groups = collections.defaultdict(list)
    for r in records:
        groups[(r["team_id"], r["cid"], r["sid"], r["role"])].append(r)
    rows = []
    # 特征键取所有记录键的并集（poss 与 def 键集不同，必须并集，否则某一侧特征整列丢失）
    META = ("team_id", "team_name", "cid", "sid", "role", "n_frames")
    feat_keys = sorted({k for r in records for k in r if k not in META})
    for (tid, cid, sid, role), recs in groups.items():
        row = {"team_id": tid, "competition_id": cid, "season_id": sid,
               "role": role, "n_frames": len(recs)}
        name = recs[0]["team_name"]
        row["team_name"] = name
        for k in feat_keys:
            vals = [r[k] for r in recs if r.get(k) is not None and not (isinstance(r[k], float) and np.isnan(r[k]))]
            row[k] = float(np.mean(vals)) if vals else np.nan
        rows.append(row)
    df = pd.DataFrame(rows)
    # pivot: 把 poss / def 两套特征合并到一行
    poss = df[df.role == "poss"].drop(columns=["role"]).add_prefix("poss_")
    deff = df[df.role == "def"].drop(columns=["role"]).add_prefix("def_")
    # 对齐键
    pk = ["team_id","competition_id","season_id"]
    poss = poss.rename(columns={f"poss_{c}": c for c in pk})
    deff = deff.rename(columns={f"def_{c}": c for c in pk})
    merged = poss.merge(deff, on=pk, how="outer", suffixes=("_poss","_def"))
    # 合并 team_name（取非空的）
    if "poss_team_name" in merged.columns and "def_team_name" in merged.columns:
        merged["team_name"] = merged["poss_team_name"].fillna(merged["def_team_name"])
        merged = merged.drop(columns=["poss_team_name","def_team_name"])
    elif "poss_team_name" in merged.columns:
        merged = merged.rename(columns={"poss_team_name":"team_name"})
    elif "def_team_name" in merged.columns:
        merged = merged.rename(columns={"def_team_name":"team_name"})
    # 丢弃全 NaN 的冗余列（来自并集特征键中某一 role 不具备的维度）
    merged = merged.dropna(axis=1, how="all")
    return merged

# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="仅处理前 N 场（调试）")
    args = ap.parse_args()

    z = zipfile.ZipFile(ZIP)
    match_index = build_match_index(z)
    t3_names = [n for n in z.namelist()
                if n.startswith("open-data-master/data/three-sixty/") and n.endswith(".json")]
    mids = [n.split("/")[-1].replace(".json", "") for n in t3_names]
    if args.limit:
        mids = mids[:args.limit]
    print(f"[360] total 360 matches: {len(mids)}")

    # 串行处理（Windows spawn 下保证使用当前源码，避免字节码/重导入陷阱）
    all_records = []
    for i, mid in enumerate(mids, 1):
        all_records.extend(process_match((mid, ZIP)))
        if i % 50 == 0:
            print(f"  processed {i} matches, frames so far={len(all_records)}")
    print(f"[360] total frames: {len(all_records)}")

    out = aggregate(all_records)
    # 仅保留两 role 都有足够样本的球队（同时有进攻与防守样本），更稳健
    out = out.dropna(subset=["poss_n_frames","def_n_frames"], how="any")
    MIN_FRAMES = 30
    out = out[(out["poss_n_frames"] >= MIN_FRAMES) & (out["def_n_frames"] >= MIN_FRAMES)]
    out = out.sort_values(["competition_id","season_id","team_name"]).reset_index(drop=True)
    fpath = os.path.join(OUTDIR, "team_features_360.csv")
    out.to_csv(fpath, index=False)
    print(f"[360] wrote {fpath}  rows={len(out)} cols={len(out.columns)}")
    print("features:", [c for c in out.columns if c not in
          ("team_id","team_name","competition_id","season_id","n_frames_poss","n_frames_def")])
    # 赛事覆盖
    cov = out.groupby(["competition_id","season_id"]).size()
    print(f"[360] team-seasons: {len(out)}  across {out[['competition_id','season_id']].drop_duplicates().shape[0]} comp-seasons")

if __name__ == "__main__":
    main()
