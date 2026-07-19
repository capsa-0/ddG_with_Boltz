# 05 — Cross-dataset transfer: Tsuboyama → FireProt

**What:** Train the raw-Δz ΔΔG predictor on the **entire Tsuboyama** corpus and test
it — with no refitting — on the independent **FireProt** dataset (real, mostly
natural proteins ≤500 aa). Every holdout in 01/06 splits *within* Tsuboyama; this is
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
- **Test:** all **3,205** FireProt mutations / **138 proteins** (≤500 aa), the *same*
  256 raw-Δz features.
- **Model:** the benchmark pipeline `SimpleImputer(median) → StandardScaler →
  estimator`, fit once on Tsuboyama and applied to FireProt. Primary model is the
  **5-seed MLP ensemble** (the project default since 06); HGB reported alongside.
- **Entry point:** `python -m ddg.evaluation.transfer --train <tsuboyama rawz>
  --test <fireprot le500 features> --model mlp`.
- **Sign convention:** FireProt (`ddG`) and Tsuboyama (`ddg`) use the **same** sign
  (positive = destabilizing; both ~75 % positive), so predictions are used as-is
  (`sign_flipped: false`). The tool auto-flips if a future pair disagrees.

## Corpus (the ≤500 FireProt set)

The FireProt test set is the union of two length bands, extracted separately (each via
prepare → predict → incremental slim → features) and merged; the bands are disjoint by
protein (a protein has one length), so the merge is a clean concat:
- **≤200 aa:** 1,543 muts / 85 proteins (`fireprot_le200`).
- **201–500 aa:** 1,662 muts / 53 proteins (`fireprot_201to500`).
- **≤500 total:** **3,205 muts / 138 proteins**, ΔΔG range [−13.7, +12.0].

(Reaching the ≤200 band's full 85 proteins required a FireProt-adapter fix — falling
back to `pdb_id` for 3 UniProt-less proteins; see `status.md`.)

## Data & provenance

| Item | Path / name |
|---|---|
| Train features | `data/processed/tsuboyama_bench_fast/rawz_features.parquet` (12,359 × 256) |
| Test configs | `experiment_configs/fireprot_le200.yaml`, `fireprot_201to500.yaml` (`mutate_first_row`, max_msa 1000, `keep_s:false`, `delete_raw:true`) |
| Test features | merged `data/processed/fireprot_le500/features_summary.parquet` (3,205 × 256) from the two bands' `features_summary.parquet` |
| Transfer code | `ddg/evaluation/transfer.py` |
| Transfer output | `data/processed/fireprot_le500/transfer_from_tsuboyama{,_hgb}/` |
| Committed tables | `transfer_summary.{csv,json}`, `transfer_summary_hgb.json`, `per_protein.csv` (this folder) |

`data/processed/` is gitignored; it lives on the cluster and as a local copy. Only
configs, raw data, and this `results/` folder are committed.

## Headline numbers (Tsuboyama → FireProt ≤500, n = 3,205 / 138 proteins)

| Metric | **MLP** | HGB |
|---|---|---|
| Pooled Pearson r | **0.648** | 0.616 |
| Pooled Spearman ρ | **0.659** | 0.631 |
| Pooled RMSE (kcal/mol) | **1.28** | 1.34 |
| Pooled MAE (kcal/mol) | **0.84** | 0.89 |
| Per-protein r — median | **0.647** | — |
| Per-protein r — mean | 0.370 | 0.419 |
| Proteins scored | 114 / 138 | 114 / 138 |

**Takeaway:** the Tsuboyama-trained raw-Δz predictor **transfers to FireProt** — a
different dataset with no shared proteins — at pooled **r ≈ 0.65 / ρ ≈ 0.66**, with a
per-protein **median r ≈ 0.65** (60 % of proteins score r > 0.5). The signal is not a
Tsuboyama artifact; it generalizes to independently curated stability data. This is
**on par with the published AFToolkit/ThermoMPNN result** (ThermoMPNN reports 0.65
Pearson transferring to FireProt from the same Megascale/Tsuboyama data with an AF2/GNN
backbone). MLP ≈ HGB (within noise), consistent with 06 — the result is carried by the
**representation**, not the estimator.

**Same known ceiling as 02/06 — magnitude, not ranking.** The model *ranks* mutations
well but **under-predicts magnitude**: the predicted-vs-measured slope is **0.27**, and
predicted ΔΔG spans only ~40 % of the measured spread (pred SD 0.66 vs measured SD 1.58).
Split by the Tsuboyama training range ([−1, 4]): **in-range r = 0.69, RMSE = 0.94**;
**out-of-range RMSE = 3.4** (n=219) with the per-tail correlation collapsing — the model
regresses toward the mean on the tails it never saw in training. Error is almost perfectly
anti-correlated with training density in ΔΔG space (Spearman ρ = −0.96). Useful for
**ranking and triage**, not absolute ΔΔG on out-of-range mutations.

## Files
- `report.pdf` — standalone narrative write-up (regenerate with `build_report.py`).
- `transfer_summary.{csv,json}` (MLP), `transfer_summary_hgb.json` (HGB) — metrics.
- `per_protein.csv` — per-protein r/ρ/RMSE/MAE/n (138 proteins).
- `figures/` — `01_transfer_scatter.png`, `02_per_protein_r_hist.png`,
  `03_error_vs_ddg.png`, `04_density_vs_error.png`; see `figures/README.md`.
- `status.md` — living log.
</content>
