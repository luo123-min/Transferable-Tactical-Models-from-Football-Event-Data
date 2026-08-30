# Transferable Tactical Models from Football Event Data

> Graph-convolutional outcome prediction and freeze-frame complementarity for
> association-football tactical modelling, built entirely on public
> [StatsBomb Open Data](https://github.com/statsbomb/open-data).

This repository contains the full, reproducible codebase for the paper
*"Transferable Tactical Models from Football Event
Data: GCN-Based Outcome Prediction and Freeze-Frame Complementarity"*  <!-- (target:
*Discover Artificial Intelligence*, Special Issue "AI-Driven Sports Science").-->

The project contributes three things:

1. **Tactical archetypes** — unsupervised clustering over hand-crafted
   passing-style / event-frequency features that generalize across 24
   competitions and both genders (922 team-seasons).
2. **Freeze-frame complementarity** — StatsBomb 360 freeze-frames quantify
   local defensive structure, pressing intensity, and passing-lane occupation;
   these features are nearly orthogonal to event-stream features, revealing a
   complementary tactical axis.
3. **A graph convolutional network (GCN)** for per-match outcome prediction,
   benchmarked against a representation-matched control (no message passing).

> **Reproducibility note.** Against the representation-matched control that
> receives the *identical* 406-dimensional node-plus-style input but performs
> no message passing, the trained GCN's macro-F1 is statistically
> indistinguishable on the held-out split (0.522 vs. 0.530; paired McNemar
> *p* = 0.63), and across five seeds the control scores higher on every seed
> (mean 0.550 vs. 0.533; paired *t*-test *p* = 0.033, Wilcoxon *p* = 0.063).
> The predictive signal resides in the representations, not in graph
> convolution per se.

---

## Repository structure

```
.
├── README.md
├── requirements.txt
├── LICENSE                  # MIT (code); data follows StatsBomb non-commercial terms
│
├── figures_paper/           # Publication figures (PDF)
│   ├── fig_pca.pdf  fig_archetypes.pdf  fig_corr360.pdf
│   ├── fig_confusion.pdf  fig_transfer.pdf
│   ├── fig_robustness.pdf  fig_ablation.pdf
│
├── results/  results_full/  results_360/  results_gnn/   # Small CSV/JSON outputs
│
├── download_data.py         # Fetch StatsBomb matches/events into data/
├── download_chunked.py      # Parallel event fetch
├── download_events.sh  download_events_par.sh  redownload_verify.sh
│
├── pipeline_v2.py           # Event-stream features + clustering (→ results_full/)
├── pipeline_360.py          # StatsBomb-360 freeze-frame features
├── pipeline_gnn.py          # Build per-match passing graphs
├── train_gnn.py             # Train GCN + MLP baselines
├── transfer_league.py       # Cross-league / cross-gender transfer + McNemar
├── validate_full.py         # Cluster validation (full event stream)
├── validate_360.py          # 360 vs event-stream complementarity
├── analyze.py               # Tactical-archetype analysis
│
├── make_figures.py          # Regenerate figures_paper/*.pdf
├── make_fig_model.py  make_fig_overview.py
├── regen_fig_ablation.py  regen_fig_robustness.py
│
└── gnn_common.py  supplementary_analysis.py  control_baseline_seeds.py   # shared GNN utils + robustness / control scripts
```

> **Note on layout.** All Python scripts live at the repository root and use
> paths relative to the repository root (and, for a few, to their own location).
> Run them from the repository root so that `data/`, `results*/`,
> `figures_paper/` resolve correctly.

---

## Environment

```bash
pip install -r requirements.txt
```

- Python ≥ 3.9
- PyTorch ≥ 1.13 (CPU is sufficient; GPU optional)
- The pipelines use only the Python standard library for data download
  (`urllib.request`, `zipfile`) — no extra credentials required.

---

## Data (do not commit)

The raw inputs are **StatsBomb Open Data**, released under a **non-commercial**
licence. They are intentionally **not** included in this repository.

1. Obtain the data:
   - **Option A (archive):** download `open-data-master.zip` from
     <https://github.com/statsbomb/open-data> and place it at the repository
     root; **or**
   - **Option B (scripts):** run `python download_data.py` (or
     `download_chunked.py`) to fetch individual match/event files into
     `data/`.
2. The `pipeline_*` and `validate_*` scripts read from `open-data-master.zip`
   (or the `data/` directory) at the repository root.

Any commercial use of the data requires a separate agreement with StatsBomb.

---

## Running the pipeline

Run from the repository root. A typical end-to-end sequence:

```bash
# 1. Features + clustering (event stream)
python pipeline_v2.py
python validate_full.py

# 2. Freeze-frame (StatsBomb 360) features and complementarity
python pipeline_360.py
python validate_360.py

# 3. Graph construction + GCN training and baselines
python pipeline_gnn.py
python train_gnn.py

# 4. Transfer experiments (cross-league / cross-gender) + significance tests
python transfer_league.py

# 5. Tactical-archetype analysis
python analyze.py

# 6. Regenerate all publication figures
python make_figures.py
python make_fig_overview.py
python make_fig_model.py
python regen_fig_ablation.py
python regen_fig_robustness.py
```

Supplementary analyses (robustness & validation):

```bash
python supplementary_analysis.py   # ablation / McNemar / leave-teams-out
python control_baseline_seeds.py      # 5-seed confirmation of the control result
```

Outputs (small CSV/JSON summaries) land in `results/`, `results_full/`,
`results_360/`, and `results_gnn/`.

---

## Licence

- **Code:** MIT — see [`LICENSE`](LICENSE).
- **Data:** StatsBomb Open Data, non-commercial licence (attribution required;
  see [`LICENSE`](LICENSE)).

---

## Citation
 <!-- 
If you use this codebase or the findings, please cite the paper
(accompanying DOI / BibTeX to be added on publication).
-->
