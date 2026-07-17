# 02 — Extrapolation to the destabilizing tail

**What:** Can the predictor extrapolate to strongly destabilizing mutations it has
never seen? Train **only on mild** mutations (|ΔΔG| < 1 kcal/mol) and test on the
**destabilizing tail** (ΔΔG > 2 kcal/mol).

**Why:** The generalization study (01) noted a predicted-vs-actual fit slope < 1 —
the model under-predicts large effects (regression to the mean). This test pushes
that to the extreme to see whether the model *interpolates* within its training
range or genuinely *extrapolates* beyond it.

**How:** HGB on 256 raw-Δz features. Fit on the mild set; evaluate on (a) held-out
mild mutations — the in-distribution baseline — and (b) the tail. Report r, RMSE,
MAE, the fit slope (`pred = a + b·true`), and range coverage.

## Result

| Split | n | r | RMSE | MAE | fit slope | pred max / true max |
|---|---|---|---|---|---|---|
| In-distribution (held-out \|ΔΔG\|<1) | 5,022 | 0.565 | 0.35 | 0.28 | 0.29 | 0.81 / 1.00 |
| **Extrapolation tail (ΔΔG>2)** | 4,435 | **0.088** | **2.28** | 2.18 | **0.02** | **0.92 / 5.68** |

**The model does not extrapolate.** On the tail the correlation is ~0 and the fit
slope is ~0.02 (flat): predictions cap at ~0.9 kcal/mol — essentially the top of
the training range — while true ΔΔG reaches 5.7. It interpolates within the ΔΔG
band it was trained on and cannot reach beyond it. Practical implication: to rank
or screen strongly destabilizing mutations, the training set must span that range.

See `extrapolation_pred_vs_actual.png` (the tail predictions are a flat cloud).

## Data & provenance

| Item | Path / name |
|---|---|
| Corpus | `tsuboyama_bench_wide` (Tsuboyama 2023, --k 90), **37,080 mutations**, 412 proteins |
| Feature table | `data/processed/tsuboyama_bench_wide/rawz_features.parquet` (256 raw-Δz cols) |
| Model / code | HGB (`ddg/evaluation/models.py`); test in `ddg/evaluation/stress.py` |
| Reproduce | `python -m ddg.evaluation.stress extrapolation --parquet <rawz_features.parquet> --out results/02_stress_extrapolation` |

## Files
- `extrapolation_pred_vs_actual.png` — in-distribution vs. tail, with y=x and fit line.
- `extrapolation_summary.csv` / `.json` — the metrics above.

Caveat: mild-train and tail-test can share proteins; this isolates ΔΔG-magnitude
extrapolation, not protein novelty (that is study 01).
