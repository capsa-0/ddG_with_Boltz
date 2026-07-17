# 03 — Learning curve (data efficiency over proteins)

**What:** How pooled accuracy scales with the number of **training proteins**, with
test proteins always held out.

**Why:** Tells us whether accuracy is data-limited (more proteins would help) or
saturated (the representation has extracted most of what it can), and how few
proteins are needed to reach useful performance.

**How:** 5-fold GroupKFold on `wt_id` (test proteins never seen in training). For
each training fraction we subsample that fraction of the *available* training
proteins, fit HGB on 256 raw-Δz features, and pool out-of-fold predictions. Each
fraction < 1.0 is averaged over 3 random protein subsamples (seeds); the shaded
error is the seed SD.

## Result

| Fraction | ~#train proteins | pooled r | pooled RMSE | pooled MAE |
|---|---|---|---|---|
| 0.10 | 33 | 0.744 | 0.668 | 0.488 |
| 0.25 | 82 | 0.767 | 0.642 | 0.466 |
| 0.50 | 165 | 0.781 | 0.626 | 0.455 |
| 1.00 | 330 | 0.793 | 0.610 | 0.444 |

**Near-saturation / high data efficiency.** Just **33 proteins** already reach
r = 0.74; a **10× increase** to 330 proteins adds only **+0.05 r** (0.744 → 0.793).
The curve is concave and flattening — the raw-Δz representation is strong enough
that the model is not badly data-starved, and adding proteins yields diminishing
returns. Seed SD is tiny (≤ 0.002), so the trend is robust.

See `learning_curve.png`.

## Data & provenance

| Item | Path / name |
|---|---|
| Corpus | `tsuboyama_bench_wide` (Tsuboyama 2023, --k 90), **37,080 mutations**, 412 proteins |
| Feature table | `data/processed/tsuboyama_bench_wide/rawz_features.parquet` (256 raw-Δz cols) |
| Model / code | HGB (`ddg/evaluation/models.py`); test in `ddg/evaluation/stress.py` |
| Reproduce | `python -m ddg.evaluation.stress learning_curve --parquet <rawz_features.parquet> --out results/03_stress_learning_curve` |

## Files
- `learning_curve.png` — pooled r (left axis) and RMSE (right axis) vs. #training proteins.
- `learning_curve.csv` — the table above, with per-fraction seed SD.
