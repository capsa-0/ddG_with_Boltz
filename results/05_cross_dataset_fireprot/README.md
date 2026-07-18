# 05 — Cross-dataset transfer: Tsuboyama → FireProt

**What:** Train the raw-Δz ΔΔG predictor on the **entire Tsuboyama** corpus and test
it — with no refitting — on the independent **FireProt** dataset (real, mostly
natural proteins ≤200 aa). Every holdout in 01/06 splits *within* Tsuboyama; this is
the first test on a **different dataset, different assay, different proteins**.

**Why:** Within-corpus generalization (01, 06) can still ride on quirks shared by one
data source (Tsuboyama's mega-scale folding assay on mostly small/designed domains).
The real question for a usable predictor is whether the Boltz raw-Δz signal carries
across to an independently curated, literature-derived stability dataset. FireProt is
a good adversary: different proteins (**zero `wt_id` overlap** with the training set),
a wider ΔΔG range ([−13.7, +12.0] vs Tsuboyama's [−2.7, +5.7]), and a different
provenance for the labels.

**How:**
- **Train:** all **12,359** Tsuboyama mutations, **256 raw-Δz features**
  (`zdiag_*` 128 + `zpool_*` 128) from `rawz_features.parquet`.
- **Test:** all **1,543** FireProt mutations / **85 proteins**, the *same* 256
  raw-Δz features from FireProt's `features_summary.parquet`.
- **Model:** the benchmark pipeline `SimpleImputer(median) → StandardScaler →
  estimator`, fit once on Tsuboyama and applied to FireProt. Primary model is the
  **5-seed MLP ensemble** (the project default since 06); HGB reported alongside.
- **Entry point:** `python -m ddg.evaluation.transfer --train <tsuboyama rawz>
  --test <fireprot features> --model mlp` (new module, commit `5e55812`).
- **Sign convention:** FireProt (`ddG`) and Tsuboyama (`ddg`) use the **same** sign
  (positive = destabilizing; both ~75 % positive), so predictions are used as-is
  (`sign_flipped: false`). The tool auto-flips if a future pair disagrees.

## Corpus completeness (why the count moved)

The FireProt feature table was rebuilt twice to reach full coverage:
- The first extraction had silently slimmed a **half-finished** predict array →
  only **773 muts / 54 proteins**.
- A clean re-run reached **1,514 / 82**, still 29 short. Root cause: the FireProt
  adapter keyed `wt_id` on `uniprot_id` alone, dropping **3 UniProt-less proteins**
  (`3PG0`/ThreeFoil, `2IMM`, `1YYX` = 29 mutations) that carry only a `pdb_id`.
- Fix (`5e55812`): `dataset_fireprot.get_wt_id` falls back to `pdb_id` →
  **1,543 muts / 85 proteins**, the full corpus used here. (2 of the 3 recovered
  proteins transfer well: `2IMM` r=0.79, `1YYX` r=0.69; `3PG0` r=−0.45.)

## Data & provenance (where everything lives)

| Item | Path / name |
|---|---|
| Train config | `experiment_configs/tsuboyama_bench_fast.yaml` |
| Train features | `data/processed/tsuboyama_bench_fast/rawz_features.parquet` (12,359 × 256) |
| Test config | `experiment_configs/fireprot_le200.yaml` (`dataset_type: fireprot`, `mutate_wt_msa`/`mutate_first_row`, max_msa 1000, `keep_s:false`, `delete_raw:true`) |
| Test raw data | `data/raw/fireprot_le200.csv` (FireProt ≤200 aa, 1,543 rows) |
| Test features | `data/processed/fireprot_le200/features_summary.parquet` (1,543 × 256) |
| Transfer code | `ddg/evaluation/transfer.py` (commit `5e55812`) |
| Transfer output (MLP) | `data/processed/fireprot_le200/transfer_from_tsuboyama/` |
| Transfer output (HGB) | `data/processed/fireprot_le200/transfer_from_tsuboyama_hgb/` |
| Committed tables | `transfer_summary.{csv,json}`, `transfer_summary_hgb.json`, `per_protein.csv` (this folder) |

`data/processed/` is gitignored; it lives on the cluster
(`/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz/data/processed/`) and as a
local copy. Only configs, raw data, and this `results/` folder are committed.

## Headline numbers (Tsuboyama → FireProt, n = 1,543 / 85 proteins)

| Metric | **MLP** | HGB |
|---|---|---|
| Pooled Pearson r | **0.621** | 0.607 |
| Pooled Spearman ρ | **0.684** | 0.675 |
| Pooled RMSE (kcal/mol) | **1.41** | 1.45 |
| Pooled MAE (kcal/mol) | **0.86** | 0.90 |
| Per-protein r — mean | 0.488 | 0.531 |
| Per-protein r — **median** | **0.668** | — |
| Proteins scored | 76 / 85 | 76 / 85 |

**Takeaway:** the Tsuboyama-trained raw-Δz predictor **transfers to FireProt** — a
different dataset with no shared proteins — at pooled **r ≈ 0.62 / ρ ≈ 0.68**, with a
per-protein **median r ≈ 0.67** (70 % of proteins score r > 0.5, 47 % > 0.7). The
signal is not a Tsuboyama artifact; it generalizes to independently curated stability
data. MLP and HGB are within noise of each other, consistent with 06 — the result is
carried by the **representation**, not the estimator.

**Same known ceiling as 02/06 — magnitude, not ranking.** The model *ranks* mutations
well but severely **under-predicts magnitude**: the predicted-vs-measured slope is
**0.26**, and predicted ΔΔG spans only ~40 % of the measured spread (pred SD 0.72 vs
measured SD 1.72). FireProt's wider range ([−13.7, +12.0]) makes this starker
than any within-Tsuboyama split — the model regresses hard toward the mean on the
destabilizing/stabilizing tails it never saw in training. Useful for **ranking and
triage**, not for absolute ΔΔG on out-of-range mutations.

## Files
- `report.pdf` — the standalone narrative write-up (regenerate with
  `python results/05_cross_dataset_fireprot/build_report.py`).
- `transfer_summary.csv` / `transfer_summary.json` — pooled + per-protein-distribution
  metrics (MLP); `transfer_summary_hgb.json` — the HGB comparison.
- `per_protein.csv` — per-protein r/ρ/RMSE/MAE/n (85 proteins).
- `figures/` — `01_transfer_scatter.png` (predicted vs measured),
  `02_per_protein_r_hist.png` (per-protein r distribution); see `figures/README.md`.
- `status.md` — living log (corpus-completeness saga, job IDs, decisions).
</content>
</invoke>
