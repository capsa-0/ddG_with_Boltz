# 13 — Balanced losses on the stabilizing tail

**What:** Does reweighting the loss toward the rare stabilizing tail fix the one
non-artifactual weakness found in `results/12_error_anatomy`?

**Why:** results/12 showed the predictor systematically calls stabilizing mutations
destabilizing (bias +0.58 kcal/mol, ρ 0.26, error 2× the class's own spread; only 4.3 %
of Tsuboyama). Constraint-aware SPURS (arXiv 2606.08100) reports S669 ρ 0.486 → 0.540
from loss-level changes alone — Balanced MSE being the term aimed at this tail. It is
the cheapest published lever: no new features, no backbone changes.

**How:** three losses, identical architecture / data / splits / seeds.
- **`mse`** — plain MSE, the results/06 baseline.
- **`bmc`** — Balanced MSE, Monte-Carlo form (Ren et al., CVPR 2022): the batch is
  treated as a classification over which target each prediction belongs to, dividing out
  the training label density. `noise_var` learned in log space.
- **`lds`** — inverse smoothed-density sample weighting over a 40-bin ΔΔG histogram.

Reimplemented in torch (`sklearn.MLPRegressor` supports neither sample weights nor a
custom objective), same `(256,128,64)` topology and antisymmetry augmentation as the
project default. 5-fold GroupKFold on `wt_id`, 2 seeds averaged, 60 epochs, batch 512.
Significance from a **cluster bootstrap over the 412 proteins** (400 resamples), reported
as the *paired* difference vs MSE so the shared resample cancels.

## Headline: it fixes the bias, not the blindness

Paired difference vs MSE, 95 % CI, held-out Tsuboyama (`*` = excludes zero):

| metric | Balanced MSE | LDS |
|---|---|---|
| **bias, stabilizing** | **−0.113 [−0.144, −0.080]** * | +0.020 [−0.011, +0.052] |
| ρ, stabilizing | +0.005 [−0.034, +0.044] | −0.008 [−0.057, +0.040] |
| AUC, stabilizing | +0.003 [−0.004, +0.011] | −0.010 [−0.022, +0.002] |
| DetPr@30 | +0.017 [−0.133, +0.167] | −0.005 [−0.167, +0.167] |
| ρ overall | +0.003 [−0.001, +0.008] | −0.011 [−0.018, −0.005] * |
| r overall | **−0.052 [−0.131, −0.001]** * | −0.042 [−0.116, +0.005] |
| MAE overall | **+0.088 [+0.076, +0.100]** * | +0.023 [+0.015, +0.034] * |

**Balanced MSE removes 19 % of the stabilizing bias (0.58 → 0.47) and improves nothing
else.** Every ranking metric on the tail — ρ, AUC, detection precision — is
statistically indistinguishable from baseline, while overall Pearson r and MAE get
significantly *worse*. **LDS is dominated**: no significant gain anywhere, significant
losses on ρ and MAE.

The figure makes the mechanism plain: BMC shifts stabilizing predictions downward
(panel 2) without changing which mutations are ranked most stabilizing (panel 3 — the
precision-vs-K curves are on top of each other).

**Interpretation.** With a **frozen** representation, loss reweighting can move
predictions but cannot create discrimination that the features do not carry. The bias is
a property of the objective and is fixable; the inability to *identify* stabilizing
mutations is a property of the representation and is not. This is consistent with
constraint-aware SPURS gaining from the same term while **fine-tuning** its backbone —
there, the representation can adapt to the reweighted objective; here it cannot.

It is also consistent with `results/11`: on S669 transfer no loss separated from any
other (r 0.37–0.39, tail ρ ≈ 0), because that error is dominated by cross-dataset domain
shift rather than by the tail.

**Recommendation: do not adopt either loss as the default.** The only real gain —
de-biased stabilizing magnitudes — costs pooled r and MAE, and does not improve the
metric that matters for engineering (finding stabilizing mutations). Revisit Balanced MSE
*after* the backbone is unfrozen, where the published precedent actually applies.

## Data & provenance

| Item | Path |
|---|---|
| Features | `data/processed/tsuboyama_bench_fast/features_ablation.parquet` |
| Transfer set | `data/processed/s669/features_ablation.parquet` |
| OOF predictions | `data/processed/_analysis/balanced_oof.csv` (gitignored) |
| Run log | `data/processed/_analysis/exp13.log` (gitignored) |
| Result tables | `results.csv` (all metrics), `bootstrap.csv` (CIs) |
| Code | `run_balanced.py` → `bootstrap.py` → `make_figure.py` |

## Figures

- `figures/01_balanced_loss.png` — paired differences with CIs; the stabilizing region
  scatter; the top-K detection-precision curves.
