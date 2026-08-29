"""
pipeline_v2.py — Unified StatsBomb feature pipeline (reads directly from the zip).

Goal: turn the full StatsBomb Open Data into ONE clean, cross-competition feature
table at the (team x competition-season) grain, ready for tactical-prototype
clustering, cross-league validation, and (later) 360 / GNN extensions.

Design choices:
- Reads JSON straight from open-data-master.zip (no 3-4 GB unpack needed).
- Grain = a team's style *within one competition-season* (stable, many samples).
- Features are orientation-agnostic where possible (robust across pitch sides):
    * passing-network topology (density, centrality, modularity, ...)
    * pass spatial profile (length, short/long/vertical fractions)
    * per-match rates of 12 event types (possession / press / defence style)
- Multi-processed over (competition_id, season_id) pairs.
- Selective: --competitions 11 2 12 9 7 16  runs only those competition ids
  (all their seasons). Omit the flag to run the whole library.

Output:
  results_full/unified_team_features.csv   (one row per team-season)
  results_full/competitions_catalog.csv    (available competitions + match counts)
"""

import zipfile
import json
import os
import argparse
import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity
from multiprocessing import Pool, cpu_count

ZIP_PATH = "open-data-master.zip"
ZIP_BASE = "open-data-master/data"
OUT_DIR = "results_full"

# Canonical event types we count as style signals (orientation-agnostic rates).
EVENT_TYPES = [
    "Pass", "Carry", "Shot", "Pressure", "Dribble", "Ball Recovery",
    "Interception", "Foul Committed", "Clearance", "Block",
    "Miscontrol", "Dispossessed",
]
EVENT_RATE_COLS = [f"rate_{t.replace(' ', '_').lower()}" for t in EVENT_TYPES]


# ----------------------------------------------------------------------------
# Zip access helpers
# ----------------------------------------------------------------------------
def _zpath(kind, *parts):
    return f"{ZIP_BASE}/{kind}/" + "/".join(str(p) for p in parts) + ".json"


def read_json(z, path):
    try:
        return json.loads(z.read(path))
    except (KeyError, json.JSONDecodeError):
        return None


# ----------------------------------------------------------------------------
# Per match -> per team raw accumulators
# ----------------------------------------------------------------------------
def process_match(z, match_id, player_team, player_position):
    """Return dict: team_id -> raw accumulation for one match."""
    evts = read_json(z, _zpath("events", match_id))
    if not evts:
        return {}

    acc = defaultdict(lambda: {
        "pass_edges": defaultdict(int),   # (src,dst) -> count
        "pass_len": [],
        "pass_dy": [],                     # |vertical component| of team-internal passes
        "pass_diag": 0,                    # count of passes where |dy| > |dx|
        "n_events": defaultdict(int),
        "n_pass_recipient_known": 0,
    })

    for ev in evts:
        team = ev.get("team", {}).get("id")
        if team is None:
            continue
        etype = ev.get("type", {}).get("name")
        if etype in EVENT_TYPES:
            acc[team]["n_events"][etype] += 1

        if etype == "Pass":
            recv = ev.get("pass", {}).get("recipient", {})
            recv_id = recv.get("id") if recv else None
            loc = ev.get("location")
            end = ev.get("pass", {}).get("end_location")
            # only count team-internal passes for the passing network
            if recv_id is not None and player_team.get(recv_id) == team:
                acc[team]["pass_edges"][(ev["player"]["id"], recv_id)] += 1
                acc[team]["n_pass_recipient_known"] += 1
                ln = ev.get("pass", {}).get("length")
                if ln is not None:
                    acc[team]["pass_len"].append(float(ln))
                # vertical/horizontal decomposition (orientation-agnostic)
                if loc and end and len(loc) >= 2 and len(end) >= 2:
                    dx = abs(float(end[0]) - float(loc[0]))
                    dy = abs(float(end[1]) - float(loc[1]))
                    acc[team]["pass_dy"].append(dy)
                    if dy > dx:
                        acc[team]["pass_diag"] += 1
            # also record length even if recipient unknown (still a style signal)
            elif ev.get("pass", {}).get("length") is not None:
                acc[team]["pass_len"].append(float(ev["pass"]["length"]))

    return acc


# ----------------------------------------------------------------------------
# Team-season aggregation
# ----------------------------------------------------------------------------
def network_features(edges):
    """Compute passing-network features from a dict of (src,dst)->weight."""
    G = nx.DiGraph()
    for (s, d), w in edges.items():
        if s == d:
            continue
        G.add_edge(s, d, weight=w)
    if G.number_of_nodes() < 2:
        return {k: 0.0 for k in [
            "net_density", "net_avg_degree", "assortativity", "transitivity",
            "n_communities", "modularity",
            "mean_degree_centrality", "mean_betweenness", "mean_closeness",
            "mean_eigenvector", "max_betweenness"]}

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
        feats["max_betweenness"] = float(max(bw.values()))
    except Exception:
        feats["mean_betweenness"] = 0.0
        feats["max_betweenness"] = 0.0
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


def team_season_features(z, comp_id, season_id, comp_meta):
    matches = read_json(z, _zpath("matches", comp_id, season_id))
    if not matches:
        return []

    rows = []
    for m in matches:
        mid = m.get("match_id")
        if mid is None:
            continue
        lineups = read_json(z, _zpath("lineups", comp_id, mid))
        player_team, player_pos = {}, {}
        if lineups:
            for side in lineups:
                tid = side.get("team_id")
                for p in side.get("lineup", []):
                    pid = p.get("player_id")
                    if pid:
                        player_team[pid] = tid
                        player_pos[pid] = (p.get("position") or {}).get("name")
        # accumulate raw per-team stats for this match
        raw = process_match(z, mid, player_team, player_pos)
        for tid, a in raw.items():
            rows.append((tid, a))

    if not rows:
        return []

    # merge across matches of this competition-season, per team
    merged = defaultdict(lambda: {"edges": defaultdict(int), "len": [], "dy": [], "diag": 0,
                                    "ev": defaultdict(int), "n_matches": 0})
    for tid, a in rows:
        m = merged[tid]
        m["n_matches"] += 1
        for (s, d), w in a["pass_edges"].items():
            m["edges"][(s, d)] += w
        m["len"].extend(a["pass_len"])
        m["dy"].extend(a["pass_dy"])
        m["diag"] += a["pass_diag"]
        for et, c in a["n_events"].items():
            m["ev"][et] += c

    out = []
    for tid, m in merged.items():
        nm = comp_meta["team_names"].get(tid, str(tid))
        n = max(m["n_matches"], 1)
        rec = {
            "team_id": tid,
            "team_name": nm,
            "competition_id": comp_id,
            "season_id": season_id,
            "competition_name": comp_meta["competition_name"],
            "season_name": comp_meta["season_name"],
            "country_name": comp_meta["country_name"],
            "competition_gender": comp_meta["competition_gender"],
            "n_matches": m["n_matches"],
            "passes_per_match": sum(m["edges"].values()) / n,
        }
        rec.update(network_features(m["edges"]))
        lens = np.array(m["len"]) if m["len"] else np.array([0.0])
        rec["mean_pass_length"] = float(np.mean(lens))
        rec["short_pass_frac"] = float(np.mean(lens < 10.0))
        rec["long_pass_frac"] = float(np.mean(lens > 30.0))
        n_int = len(m["dy"])  # team-internal passes with location data
        rec["vertical_pass_frac"] = float(np.mean(m["dy"])) if m["dy"] else 0.0
        rec["diag_pass_frac"] = (m["diag"] / n_int) if n_int else 0.0
        # event rates per match
        for et, col in zip(EVENT_TYPES, EVENT_RATE_COLS):
            rec[col] = m["ev"].get(et, 0) / n
        out.append(rec)
    return out


# ----------------------------------------------------------------------------
# Worker wrapper for multiprocessing
# ----------------------------------------------------------------------------
def _init_worker(zip_path):
    global _Z
    _Z = zipfile.ZipFile(zip_path)


def _worker(task):
    comp_id, season_id, meta = task
    return team_season_features(_Z, comp_id, season_id, meta)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def build_catalog(z):
    comps = json.loads(z.read(f"{ZIP_BASE}/competitions.json"))
    rows = []
    for c in comps:
        cid, sid = c["competition_id"], c["season_id"]
        matches = read_json(z, _zpath("matches", cid, sid))
        rows.append({
            "competition_id": cid,
            "season_id": sid,
            "competition_name": c["competition_name"],
            "season_name": c["season_name"],
            "country_name": c["country_name"],
            "competition_gender": c["competition_gender"],
            "n_matches": len(matches) if matches else 0,
        })
    return comps, pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--competitions", nargs="*", type=int, default=None,
                    help="restrict to these competition_ids (all seasons). Omit = whole library.")
    ap.add_argument("--workers", type=int, default=max(1, cpu_count() - 1))
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    z = zipfile.ZipFile(ZIP_PATH)
    comps, catalog = build_catalog(z)
    catalog.to_csv(os.path.join(OUT_DIR, "competitions_catalog.csv"), index=False)
    print(f"[catalog] {len(comps)} competitions written")

    # build task list
    comp_meta = {}
    for c in comps:
        cid, sid = c["competition_id"], c["season_id"]
        teams = {}
        # team names come from matches files; lazily fill later via matches
        comp_meta[(cid, sid)] = {
            "competition_name": c["competition_name"],
            "season_name": c["season_name"],
            "country_name": c["country_name"],
            "competition_gender": c["competition_gender"],
            "team_names": {},
        }
    if args.competitions:
        wanted = set(args.competitions)
        tasks = [(cid, sid, comp_meta[(cid, sid)]) for (cid, sid) in comp_meta if cid in wanted]
    else:
        tasks = [(cid, sid, m) for (cid, sid), m in comp_meta.items()]
    print(f"[tasks] {len(tasks)} competition-seasons to process")

    # enrich team names from matches (cheap, one read per comp-season)
    for cid, sid, _ in tasks:
        ms = read_json(z, _zpath("matches", cid, sid))
        if not ms:
            continue
        for m in ms:
            for side in ("home_team", "away_team"):
                t = m.get(side)
                if t:
                    tid = t.get("home_team_id") if side == "home_team" else t.get("away_team_id")
                    tnm = t.get("home_team_name") if side == "home_team" else t.get("away_team_name")
                    if tid is not None:
                        comp_meta[(cid, sid)]["team_names"][tid] = tnm

    all_rows = []
    if args.workers > 1 and len(tasks) > 1:
        with Pool(processes=args.workers, initializer=_init_worker, initargs=(ZIP_PATH,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_worker, tasks), 1):
                all_rows.extend(res)
                if i % 10 == 0:
                    print(f"  processed {i}/{len(tasks)} comp-seasons, {len(all_rows)} team-seasons so far")
    else:
        _init_worker(ZIP_PATH)
        for i, t in enumerate(tasks, 1):
            all_rows.extend(_worker(t))
            if i % 10 == 0:
                print(f"  processed {i}/{len(tasks)} comp-seasons, {len(all_rows)} team-seasons so far")

    df = pd.DataFrame(all_rows)
    feat_cols = [c for c in df.columns if c not in (
        "team_id", "team_name", "competition_id", "season_id",
        "competition_name", "season_name", "country_name",
        "competition_gender", "n_matches")]
    out_csv = os.path.join(OUT_DIR, "unified_team_features.csv")
    df.to_csv(out_csv, index=False)
    print(f"[done] {len(df)} team-seasons, {len(feat_cols)} feature cols -> {out_csv}")
    print("competitions covered:", df["competition_name"].nunique(),
          "| teams:", df["team_name"].nunique(),
          "| genders:", df["competition_gender"].unique().tolist())


if __name__ == "__main__":
    main()
