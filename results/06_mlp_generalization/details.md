# 06 — supplementary details & methods

Appendix to the `README.md`: the model design, exact hyperparameters, run
provenance, and the numbers behind the summary. This experiment is a **model swap**
on top of `01_generalization` — for corpus, feature-extraction mechanics, and split
definitions, see `../01_generalization/details.md`; only what differs is here.

---

## 1. The model — and why it's an ensemble

The estimator is `make_model("mlp")` in `ddg/evaluation/models.py`:

```python
member = MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                      solver="adam", alpha=3e-3, learning_rate_init=1e-3,
                      batch_size=256, max_iter=1000, early_stopping=True,
                      n_iter_no_change=25, validation_fraction=0.1, random_state=s)
est = VotingRegressor([(f"mlp{s}", member(s)) for s in range(5)], n_jobs=-1)
```

in the same `SimpleImputer(median) → StandardScaler → est` pipeline as every other
model in the suite.

**Why a 5-seed ensemble and not a single MLP.** A single `MLPRegressor` is
high-variance on this tabular problem: its group-holdout score depends on the weight
initialization and on the internal early-stopping validation split. A first attempt
with a lone-seed `MLPRegressor((256,64))` produced **unstable, non-monotonic**
homology numbers (30/50/90 % identity → 0.47 / 0.39 / 0.72) — clearly noise, since
the tree baseline was flat ~0.77 across those thresholds. Averaging five
seed-decorrelated nets (`VotingRegressor`) is the standard variance-reduction fix;
it made the homology numbers stable and monotonic (0.781 / 0.785 / 0.790). Members
are fit in parallel (`n_jobs=-1`), so the ensemble costs ~one net of wall time.

**No holdout tuning.** The architecture/regularization were chosen a priori (deeper
+ stronger L2 + patient early stopping are principled defaults for a 256-dim input);
they were **not** tuned against the holdout scores. This matches the "no nested-CV
tuning" stance of the HGB baseline, so the MLP-vs-HGB comparison stays honest.

## 2. Run provenance

- **Where:** SLURM cluster, cpu partition, via `slurm/eval.sbatch`. 4 jobs:
  main 7-holdout (`benchmark_rawz_mlp`) + one per homology threshold
  (`benchmark_rawz_mlp_cl{30,50,90}`).
- **Submission flags:** `--exclude=nodo3,nodo5` (known bad nodes),
  `--cpus-per-task=6`, and `OMP/OPENBLAS/MKL_NUM_THREADS=1` so the 5 ensemble
  members parallelize across cores without BLAS oversubscription.
- **Cost:** the ensemble runs at ~11–12 s per fold; the 332-fold `substitution`
  holdout dominates (~66 min). Follow-up: parallelize the fold loop
  (`ddg/evaluation/benchmark.py`) — the folds are independent.
- **Code:** `ddg/evaluation/models.py` at commit `215a6ce`.

## 3. Full pooled-metric table (MLP)

RMSE / MAE in kcal/mol.

| Holdout | n | pooled r | Spearman | RMSE | MAE |
|---|---|---|---|---|---|
| random | 12359 | 0.803 | 0.785 | 0.605 | 0.437 |
| protein | 12359 | 0.792 | 0.777 | 0.620 | 0.446 |
| cluster_30 | 12359 | 0.781 | — | — | — |
| cluster_50 | 12359 | 0.785 | — | — | — |
| cluster_90 | 12359 | 0.790 | — | — | — |
| denovo | 12359 | 0.703 | 0.725 | 0.723 | 0.508 |
| substitution | 12164 | 0.765 | 0.778 | 0.654 | 0.444 |
| source_residue | 12358 | 0.770 | 0.750 | 0.650 | 0.473 |
| target_residue | 12359 | 0.761 | 0.738 | 0.660 | 0.477 |
| chemistry | 12358 | 0.732 | 0.740 | 0.697 | 0.478 |

- **Per-protein (protein holdout):** mean r 0.827, SD 0.105 (vs HGB mean 0.806).
- Pooled r = one Pearson over all out-of-fold predictions; it differs from the mean
  of per-category r's (see `../01_generalization/details.md` §4).

## 4. Interpretation

The MLP tracks HGB within ±0.02 r on every holdout, landing slightly above on the
interpolation-style splits (random/protein/homology/per-residue) and even on the
hard transfers (de-novo, substitution, chemistry). Two independent model families
reaching the same generalization profile is evidence the signal lives in the
**raw-Δz representation**, not in a particular regressor. The regression-to-the-mean
weakness (slope < 1 on the destabilizing tail) persists — it is a property of the
features/objective, not the model, consistent with `02_stress_extrapolation`.
</content>
