"""
pipeline_gnn.py — Build GNN-ready passing-network graphs from the full
StatsBomb Open Data library, for the trained-GNN extension (option ③).

For every match we build ONE graph per team (the team's passing network):
  * nodes  = players who passed to / received from a team-mate
  * edges  = weighted by pass count between player pairs (team-internal)
  * node features (16-dim): position one-hot(4) + out/in degree share +
                   pass-success rate + home/away flag + starter flag +
                   late-pass share (time-decay) + pass-type shares
                   (short/long/cross/through) + avg pass length + connectivity
  * adjacency = symmetrically-normalised (A+I) with self-loops
  * label  = that team's match result (loss=0 / draw=1 / win=2)

We also emit, per graph, a set of hand-crafted aggregate features (the kind
used by the unsupervised prototype) so the trained GNN can be compared
fairly against a classic ML baseline on the SAME label grain.

Reads directly from open-data-master.zip (no unpack needed).
Outputs:
  results_gnn/graph_data.npz
      node_feat [G, MAXN, F]   float32
      adj       [G, MAXN, MAXN] float32  (normalised, with self-loops)
      mask      [G, MAXN]       float32  (1 = real node)
      label     [G]             int64    (0/1/2)
      handcrafted [G, H]        float32
  results_gnn/graph_meta.csv    (one row per graph, for split / plotting)
"""

import zipfile
import json
import os
import argparse
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity

ZIP_PATH = "open-data-master.zip"
ZIP_BASE = "open-data-master/data"
OUT_DIR = "results_gnn"
MAXN = 20          # pad player count to this (covers 11 + subs)
# node feature dim (16): pos onehot(4) + out + in + success + is_home
#   + is_starter + late_pass_share(time-decay) + pass-type shares(short/long/
#   cross/through) + avg_pass_length + connectivity
F = 16
POS_BUCKETS = ["GK", "DF", "MF", "FW"]
LATE_MINUTE = 70    # passes at minute >= this count as "late" (time-decay proxy)

EVENT_TYPES = ["Pass", "Carry", "Shot", "Pressure", "Dribble",
               "Ball Recovery", "Interception", "Clearance"]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _zpath(kind, *parts):
    return f"{ZIP_BASE}/{kind}/" + "/".join(str(p) for p in parts) + ".json"


def read_json(z, path):
    try:
        return json.loads(z.read(path))
    except (KeyError, json.JSONDecodeError):
        return None


def pos_bucket(name):
    if not name:
        return 2
    n = name.lower()
    if "goalkeeper" in n:
        return 0
    if "back" in n or "defender" in n:
        return 1
    if "midfield" in n:
        return 2
    if "wing" in n or "forward" in n or "striker" in n or "winger" in n:
        return 3
    return 2


def network_features(edges):
    """Passing-network topology from a dict of (src,dst)->weight."""
    G = nx.DiGraph()
    for (s, d), w in edges.items():
        if s == d:
            continue
        G.add_edge(s, d, weight=w)
    if G.number_of_nodes() < 2:
        return dict(net_density=0.0, net_avg_degree=0.0, assortativity=0.0,
                    transitivity=0.0, n_communities=1, modularity=0.0,
                    mean_degree_centrality=0.0, mean_betweenness=0.0,
                    mean_closeness=0.0, mean_eigenvector=0.0)
    Gu = G.to_undirected()
    feats = {}
    feats["net_density"] = nx.density(G)
    feats["net_avg_degree"] = float(np.mean([d for _, d in G.degree()]))
    try:
        feats["assortativity"] = nx.degree_assortativity_coefficient(G)
    except Exception:
        feats["assortativity"] = 0.0
    try:
        feats["transitivity"] = nx.transitivity(Gu)
    except Exception:
        feats["transitivity"] = 0.0
    try:
        comm = greedy_modularity_communities(Gu, weight="weight")
        feats["n_communities"] = len(comm)
        feats["modularity"] = modularity(Gu, comm, weight="weight")
    except Exception:
        feats["n_communities"] = 1
        feats["modularity"] = 0.0
    try:
        feats["mean_degree_centrality"] = float(np.mean(list(nx.degree_centrality(G).values())))
    except Exception:
        feats["mean_degree_centrality"] = 0.0
    try:
        bw = nx.betweenness_centrality(G, weight="weight")
        feats["mean_betweenness"] = float(np.mean(list(bw.values())))
    except Exception:
        feats["mean_betweenness"] = 0.0
    try:
        feats["mean_closeness"] = float(np.mean(list(nx.closeness_centrality(G).values())))
    except Exception:
        feats["mean_closeness"] = 0.0
    try:
        ev = nx.eigenvector_centrality_numpy(Gu, weight="weight")
        feats["mean_eigenvector"] = float(np.mean(list(ev.values())))
    except Exception:
        feats["mean_eigenvector"] = 0.0
    return feats


# ----------------------------------------------------------------------------
# per-match graph builder
# ----------------------------------------------------------------------------
def process_match(z, mid, mmeta, comp_meta):
    """Return list of graph records (one per team) for this match, or []."""
    lineups = read_json(z, _zpath("lineups", mid))
    evts = read_json(z, _zpath("events", mid))
    if not evts or not lineups:
        return []

    # player -> (team_id, position bucket)
    pteam, ppos = {}, {}
    for side in lineups:
        tid = side.get("team_id")
        for p in side.get("lineup", []):
            pid = p.get("player_id")
            if pid is not None:
                pteam[pid] = tid
                ppos[pid] = pos_bucket((p.get("position") or {}).get("name"))

    # per-team accumulators
    teams = {mmeta["home_team_id"], mmeta["away_team_id"]}
    acc = {t: {"edges": defaultdict(int),
               "outdeg": defaultdict(int), "indeg": defaultdict(int),
               "comp": defaultdict(int), "fail": defaultdict(int),
               "plen": [], "dy": 0, "dxdy": 0,
               "ev": defaultdict(int),
               # per-player type / time / connectivity features (node upgrade)
               "ptot": defaultdict(int), "plate": defaultdict(int),
               "pshort": defaultdict(int), "plong": defaultdict(int),
               "pcross": defaultdict(int), "pthru": defaultdict(int),
               "pfirst": {}, "pteammates": defaultdict(set)} for t in teams}

    for ev in evts:
        team = ev.get("team", {}).get("id")
        if team not in acc:
            continue
        etype = ev.get("type", {}).get("name")
        if etype in EVENT_TYPES:
            acc[team]["ev"][etype] += 1
        if etype != "Pass":
            continue
        pid = ev.get("player", {}).get("id")
        recv = ev.get("pass", {}).get("recipient", {})
        recv_id = recv.get("id") if recv else None
        if pid is None or recv_id is None:
            continue
        # team-internal pass only
        if pteam.get(recv_id) != team:
            continue
        a = acc[team]
        minute = ev.get("minute")
        ptype_name = ev.get("pass", {}).get("type", {}).get("name")
        ptechnique = ev.get("pass", {}).get("technique", {}).get("name")
        is_cross = (ptype_name == "Cross")
        is_thru = (ptype_name == "Through Ball") or (ptechnique == "Through Ball")
        ln = ev.get("pass", {}).get("length")
        is_long = (ln is not None and ln > 30)
        is_short = (ln is not None and ln <= 30)
        a["edges"][(pid, recv_id)] += 1
        a["outdeg"][pid] += 1
        a["indeg"][recv_id] += 1
        if ev.get("pass", {}).get("outcome") is None:
            a["comp"][pid] += 1
        else:
            a["fail"][pid] += 1
        if ln is not None:
            a["plen"].append(float(ln))
        loc = ev.get("location")
        end = ev.get("pass", {}).get("end_location")
        if loc and end and len(loc) >= 2 and len(end) >= 2:
            dx = abs(float(end[0]) - float(loc[0]))
            dy = abs(float(end[1]) - float(loc[1]))
            a["dy"] += dy
            if dy > dx:
                a["dxdy"] += 1
        # per-player type / time / connectivity features (node upgrade)
        a["ptot"][pid] += 1
        if minute is not None and minute >= LATE_MINUTE:
            a["plate"][pid] += 1
        a["pshort"][pid] += 1 if is_short else 0
        a["plong"][pid] += 1 if is_long else 0
        a["pcross"][pid] += 1 if is_cross else 0
        a["pthru"][pid] += 1 if is_thru else 0
        if a["pfirst"].get(pid) is None and minute is not None:
            a["pfirst"][pid] = minute
        a["pteammates"][pid].add(recv_id)

    records = []
    hs, as_ = mmeta["home_score"], mmeta["away_score"]
    for t in teams:
        a = acc[t]
        total_edges = sum(a["edges"].values())
        nodes = set()
        for (s, d) in a["edges"]:
            nodes.add(s); nodes.add(d)
        if len(nodes) < 3 or total_edges < 5:
            continue
        # sort nodes by total degree, keep top MAXN
        deg = {n: a["outdeg"][n] + a["indeg"][n] for n in nodes}
        ordered = sorted(nodes, key=lambda n: -deg[n])[:MAXN]
        idx = {n: i for i, n in enumerate(ordered)}
        n_real = len(ordered)

        # node features (16-dim, upgraded)
        nf = np.zeros((MAXN, F), dtype=np.float32)
        avg_len = (float(np.mean(a["plen"])) / 50.0) if a["plen"] else 0.0
        for i, n in enumerate(ordered):
            b = ppos.get(n, 2)
            nf[i, b] = 1.0                                   # 0-3 position one-hot
            nf[i, 4] = a["outdeg"][n] / max(total_edges, 1)   # out-degree share
            nf[i, 5] = a["indeg"][n] / max(total_edges, 1)    # in-degree share
            c, f = a["comp"][n], a["fail"][n]
            nf[i, 6] = (c / (c + f)) if (c + f) > 0 else 0.0  # pass success rate
            nf[i, 7] = 1.0 if t == mmeta["home_team_id"] else 0.0  # is_home
            tot = a["ptot"][n]
            nf[i, 8] = 1.0 if (a["pfirst"].get(n) is not None and a["pfirst"][n] <= 15) else 0.0  # starter
            nf[i, 9] = (a["plate"][n] / tot) if tot > 0 else 0.0   # late-pass share (time decay)
            sh, lo, cr, th = a["pshort"][n], a["plong"][n], a["pcross"][n], a["pthru"][n]
            s = sh + lo + cr + th
            nf[i, 10] = (sh / s) if s > 0 else 0.0           # short-pass share
            nf[i, 11] = (lo / s) if s > 0 else 0.0           # long-pass share
            nf[i, 12] = (cr / s) if s > 0 else 0.0           # cross/through? -> cross
            nf[i, 13] = (th / s) if s > 0 else 0.0           # through-ball share
            nf[i, 14] = avg_len                                # avg pass length (team-level)
            nf[i, 15] = (len(a["pteammates"][n]) / max(1, n_real - 1))  # connectivity

        # adjacency: BINARY edges (structure only) + self-loop, then the
        # standard GCN renormalisation  D^{-1/2}(A+I)D^{-1/2}.  Pass volume is
        # already carried by the node degree features, so we keep the adjacency
        # binary to avoid heavy edges dominating the renormalisation.
        A = np.zeros((MAXN, MAXN), dtype=np.float32)
        for (s, d) in a["edges"]:
            if s in idx and d in idx:
                A[idx[s], idx[d]] = 1.0
                A[idx[d], idx[s]] = 1.0
        for i in range(MAXN):
            A[i, i] = 1.0          # self-loop so every node has degree >= 1
        deg_row = A.sum(axis=1)
        deg_row = np.where(deg_row > 0, deg_row, 1.0)
        Dinv = np.diag(1.0 / np.sqrt(deg_row))
        Anorm = Dinv @ A @ Dinv
        mask = np.zeros(MAXN, dtype=np.float32)
        mask[:n_real] = 1.0

        # label: this team's result
        if t == mmeta["home_team_id"]:
            label = 2 if hs > as_ else (1 if hs == as_ else 0)
        else:
            label = 2 if as_ > hs else (1 if hs == as_ else 0)

        # hand-crafted aggregate features (mirror of unsupervised prototype)
        nf2 = network_features(a["edges"])
        plens = np.array(a["plen"]) if a["plen"] else np.array([0.0])
        comp_total = sum(a["comp"].values())
        fail_total = sum(a["fail"].values())
        hand = [
            float(total_edges),
            (comp_total / (comp_total + fail_total)) if (comp_total + fail_total) > 0 else 0.0,
            float(np.mean(plens)),
            float(np.mean(plens < 10.0)),
            float(np.mean(plens > 30.0)),
            (a["dy"] / len(a["plen"])) if a["plen"] else 0.0,
            nf2["net_density"], nf2["net_avg_degree"], nf2["assortativity"],
            nf2["transitivity"], float(nf2["n_communities"]), nf2["modularity"],
            nf2["mean_degree_centrality"], nf2["mean_betweenness"],
            nf2["mean_closeness"], nf2["mean_eigenvector"],
            float(a["ev"].get("Pass", 0)), float(a["ev"].get("Carry", 0)),
            float(a["ev"].get("Shot", 0)), float(a["ev"].get("Pressure", 0)),
            float(a["ev"].get("Dribble", 0)),
            1.0 if t == mmeta["home_team_id"] else 0.0,
        ]

        records.append({
            "node_feat": nf, "adj": Anorm, "mask": mask,
            "label": label, "handcrafted": np.array(hand, dtype=np.float32),
            "meta": {
                "match_id": mid, "team_id": t,
                "team_name": comp_meta["team_names"].get(t, str(t)),
                "competition_name": comp_meta["competition_name"],
                "season_name": comp_meta["season_name"],
                "country_name": comp_meta["country_name"],
                "competition_gender": comp_meta["competition_gender"],
                "is_home": 1 if t == mmeta["home_team_id"] else 0,
                "home_score": hs, "away_score": as_,
            },
        })
    return records


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="cap number of matches processed (debug). Omit = full library.")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    z = zipfile.ZipFile(ZIP_PATH)
    comps = json.loads(z.read(f"{ZIP_BASE}/competitions.json"))

    # build match metadata + team names
    match_meta = {}     # mid -> dict
    comp_meta = {}      # (cid,sid) -> meta dict
    for c in comps:
        cid, sid = c["competition_id"], c["season_id"]
        cm = {
            "competition_name": c["competition_name"],
            "season_name": c["season_name"],
            "country_name": c["country_name"],
            "competition_gender": c["competition_gender"],
            "team_names": {},
        }
        ms = read_json(z, _zpath("matches", cid, sid))
        if not ms:
            comp_meta[(cid, sid)] = cm
            continue
        for m in ms:
            mid = m.get("match_id")
            if mid is None:
                continue
            h = m.get("home_team", {})
            a = m.get("away_team", {})
            match_meta[mid] = {
                "comp_id": cid, "sid": sid,
                "home_team_id": h.get("home_team_id"),
                "away_team_id": a.get("away_team_id"),
                "home_score": (m.get("home_score") if m.get("home_score") is not None else -1),
                "away_score": (m.get("away_score") if m.get("away_score") is not None else -1),
            }
            cm["team_names"][h.get("home_team_id")] = h.get("home_team_name")
            cm["team_names"][a.get("away_team_id")] = a.get("away_team_name")
        comp_meta[(cid, sid)] = cm

    # drop matches with no usable score
    mids = [mid for mid, mm in match_meta.items()
            if mm["home_score"] >= 0 and mm["away_score"] >= 0
            and mm["home_team_id"] is not None and mm["away_team_id"] is not None]
    if args.limit:
        mids = mids[:args.limit]
    print(f"[setup] {len(mids)} matches with valid scores")

    all_recs = []
    skipped = 0
    for i, mid in enumerate(mids, 1):
        mm = match_meta[mid]
        cm = comp_meta[(mm["comp_id"], mm["sid"])]
        recs = process_match(z, mid, mm, cm)
        if not recs:
            skipped += 1
        all_recs.extend(recs)
        if i % 500 == 0:
            print(f"  {i}/{len(mids)} matches, {len(all_recs)} graphs so far")

    print(f"[done] {len(all_recs)} team-match graphs from {len(mids)-skipped} matches")

    node_feat = np.stack([r["node_feat"] for r in all_recs])
    adj = np.stack([r["adj"] for r in all_recs])
    mask = np.stack([r["mask"] for r in all_recs])
    label = np.array([r["label"] for r in all_recs], dtype=np.int64)
    handcrafted = np.stack([r["handcrafted"] for r in all_recs])
    meta = pd.DataFrame([r["meta"] for r in all_recs])

    np.savez_compressed(
        os.path.join(OUT_DIR, "graph_data.npz"),
        node_feat=node_feat, adj=adj, mask=mask, label=label,
        handcrafted=handcrafted,
    )
    meta.to_csv(os.path.join(OUT_DIR, "graph_meta.csv"), index=False)

    print(f"[saved] graphs={node_feat.shape}  adj={adj.shape}  labels="
          f"{dict(zip(*np.unique(label, return_counts=True)))}")
    print(f"  competitions={meta['competition_name'].nunique()} "
          f"teams={meta['team_name'].nunique()} "
          f"genders={meta['competition_gender'].unique().tolist()}")


def _sid_of(comps, cid, mid):
    """Recover season_id for a match (matches file is per comp-season)."""
    return _MID_SID.get(mid, None)


# precompute mid->sid globally for _sid_of
_MID_SID = {}


if __name__ == "__main__":
    main()
