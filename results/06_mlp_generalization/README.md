# 06 — Generalization holdouts with an MLP (raw-Δz predictor)

**What:** A model-swap replication of [`01_generalization`](../01_generalization/):
the exact same corpus, features, and holdout suite, but the regressor is a
**neural-network MLP** instead of `HistGradientBoosting`. Answers "does the raw-Δz
generalization result depend on the model family, or on the representation?"

**Why:** Experiment 01 established that a raw-Δz feature set generalizes across
proteins/homology/de-novo, using gradient-boosted trees. If an MLP reaches the same
numbers, the result is a property of the **representation**, not of one estimator —
and we get a second, independent model to fall back on.

**How:**
- **Corpus / features:** identical to 01 — Tsuboyama `tsuboyama_bench_fast`, **12,359
  mutations**, **256 raw-Δz features** (`Δz_diagonal` 128 + `Δz_row-pooled` 128),
  from `rawz_features.parquet`.
- **Model:** a **5-seed `VotingRegressor` of MLPs** in the same
  `SimpleImputer(median) → StandardScaler → estimator` pipeline. Each member is
  `MLPRegressor(hidden=(256,128,64), alpha=3e-3, early_stopping, n_iter_no_change=25,
  max_iter=1000)`; the five are averaged. See `details.md` for why it's an ensemble
  (a single-seed MLP was unstable across group folds).
- **Suite:** `python -m ddg.evaluation --model mlp` (splits in
  `ddg/evaluation/splits.py`); homology sweep run per identity threshold with
  `--holdouts cluster --cluster-map cluster_map_{30,50,90}.csv`.

## Data & provenance (where everything lives)

| Item | Path / name |
|---|---|
| Experiment config | `experiment_configs/tsuboyama_bench_fast.yaml` |
| Feature table | `data/processed/tsuboyama_bench_fast/rawz_features.parquet` (256 raw-Δz cols) |
| Model / eval code | `ddg/evaluation/` — `make_model("mlp")` in `models.py` (commit `215a6ce`) |
| Benchmark output (main) | `data/processed/tsuboyama_bench_fast/benchmark_rawz_mlp/` |
| Benchmark output (homology) | `…/benchmark_rawz_mlp_cl{30,50,90}/` |
| Committed summary | `benchmark_summary.csv` (this folder — 7 holdouts + 3 homology rows) |

`data/processed/` is gitignored; it lives on the cluster
(`/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz/data/processed/`) and as a
local copy. Only the config and this `results/` folder are committed.

## Headline numbers (pooled Pearson r) — MLP vs the HGB baseline (01)

| Holdout | HGB (01) | **MLP** | Δ |
|---|---|---|---|
| Random (10-fold) | 0.783 | **0.803** | +0.020 |
| Protein (GroupKFold on wt_id) | 0.774 | **0.792** | +0.018 |
| Homology 30 / 50 / 90 % | 0.765 / 0.766 / 0.772 | **0.781 / 0.785 / 0.790** | +~0.017 |
| De-novo (natural ↔ designed) | 0.705 | **0.703** | −0.002 |
| Substitution (leave-one-out) | 0.772 | **0.765** | −0.007 |
| Source residue (X→*) | 0.754 | **0.770** | +0.016 |
| Target residue (*→X) | 0.743 | **0.761** | +0.018 |
| Chemistry class | 0.734 | **0.732** | −0.002 |
| Per-protein mean r | 0.806 | **0.827** | +0.021 |

**Takeaway:** the MLP **matches or slightly beats** HGB on every holdout — a touch
ahead on the interpolation-style splits (random, protein, homology, per-residue) and
statistically even on the hardest transfers (de-novo, substitution, chemistry). The
generalization result is a property of the **raw-Δz representation**, not of the
tree model; the MLP is a viable interchangeable regressor.

Same known weakness as 01: the predicted-vs-actual slope is < 1 (under-prediction of
the most destabilizing mutations); the model interpolates, it does not extrapolate.

## Files
- `benchmark_summary.csv` — the numbers above (7 holdouts + 3 homology thresholds).
- `details.md` — model design (why an ensemble), hyperparameters, provenance, the
  per-number appendix.
- `figures/` — numbered PNGs; see `figures/README.md`.
</content>
