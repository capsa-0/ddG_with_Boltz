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
