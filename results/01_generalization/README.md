# 01 — Generalization holdouts (raw-Δz predictor)

**What:** How well the ΔΔG predictor generalizes across increasingly strict
train/test splits, using the decided **raw-Δz** feature representation.

**Why:** Random CV overstates real-world performance because mutations of the same
protein leak between folds. This study measures the drop as we hold out whole
proteins, homology clusters, de-novo vs. natural proteins, and individual
substitutions/residues.

**How:**
- **Corpus:** Tsuboyama et al. 2023, `tsuboyama_bench_fast` — all 412 proteins,
  ~30 mutations each, **12,359 mutations**.
- **Features:** raw Δz = `Δz_diagonal` (128) + `Δz_row-pooled` (128) = **256**
  (`mut − wt` at the mutated residue; no `s`, no summary stats).
- **Model:** `HistGradientBoostingRegressor` in a
  `SimpleImputer(median) → StandardScaler → HGB` pipeline.
- **Suite:** `python -m ddg.evaluation` (splits in `ddg/evaluation/splits.py`).

## Data & provenance (where everything lives)

| Item | Path / name |
|---|---|
| Experiment config | `experiment_configs/tsuboyama_bench_fast.yaml` |
| Source dataset | Tsuboyama et al. 2023 single mutants, `data/raw/tsuboyama_single_mutants_ddg.csv` (389k rows, 412 proteins) |
| This corpus (subset) | `data/raw/tsuboyama_bench_fast.csv` — built by `ddg_datasets/build_benchmark_corpus.py --k 30` (all 412 proteins, ~30 muts each, stratified) |
| Processed dir | `data/processed/tsuboyama_bench_fast/` (mutations.csv, slim store, features) |
| Feature table | `rawz_features.parquet` (256 raw-Δz cols) — equivalently now `features_summary.parquet` from the refactored `features` step |
| Benchmark output | `data/processed/tsuboyama_bench_fast/benchmark_rawz/` (`benchmark_summary.csv`, `per_unit/`, `figures/`) |
| Model / eval code | `ddg/evaluation/` (`python -m ddg.evaluation --config experiment_configs/tsuboyama_bench_fast.yaml`) |

Note: `data/processed/` is gitignored — it lives on the cluster
(`/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz/data/processed/`) and as
a local copy; only the config and this `results/` folder are committed.

## Headline numbers (pooled Pearson r)

| Holdout | r |
|---|---|
| Random (10-fold) | 0.783 |
| Protein (GroupKFold on wt_id) | 0.702 |
| Homology 30 / 50 / 90 % identity | 0.765 / 0.766 / 0.772 |
| De-novo (natural ↔ designed) | 0.615 |
| Per-protein mean r | 0.806 (median 0.831) |

Known weakness: the predicted-vs-actual fit slope is < 1 — the model
under-predicts the most destabilizing mutations (regression to the mean).

## Files
- `report.pdf` — the narrative report (read first).
- `details.md` — exact mechanics, hyperparameters, provenance, and the numbers
  behind the summary statements.
- `figures/` — numbered PNGs; see `figures/README.md` for the index.
