# Status — 13_balanced_loss

**State:** ✅ Done
**Last updated:** 2026-08-25

## Current state

Direct follow-up to `results/12_error_anatomy`, which found the predictor's only
non-artifactual weakness: **stabilizing mutations** (4.3 % of Tsuboyama; bias +0.56,
ρ 0.27, MAE 2× the class's own spread). Constraint-aware SPURS (arXiv 2606.08100)
reports S669 ρ 0.486 → 0.540 from loss-level changes alone, of which **Balanced MSE** is
the term aimed at exactly this tail — so this tests whether the same trick transfers to
Boltz-embedding features.

Three losses, identical architecture / data / splits / seeds:
- **`mse`** — plain MSE, the results/06 baseline.
- **`bmc`** — Balanced MSE, Monte-Carlo form (Ren et al., CVPR 2022): the batch is
  treated as a classification over which target each prediction belongs to, which
  divides out the training label density. `noise_var` learned in log space.
- **`lds`** — inverse smoothed-density sample weighting over a 40-bin ΔΔG histogram.

Reimplemented in **torch** (CPU) because `sklearn.MLPRegressor` supports neither sample
weights nor a custom loss. Same topology as the project default: `(256,128,64)`, ReLU,
AdamW `lr 1e-3 / wd 1e-4`, cosine schedule, antisymmetry augmentation. 5-fold GroupKFold
on `wt_id`, 2 seeds averaged, 60 epochs, batch 512 (a large batch matters for BMC — it
is the Monte-Carlo sample for the density estimate).

Evaluated on **held-out Tsuboyama (OOF)** and, using the same trained models,
**transferred to S669**. Reports overall *and* stabilizing-subset metrics
(`stab_rho`, `stab_bias`, `auc_stab`, `detpr30`, `ndcg30`) — a tail method must not be
judged on pooled correlation alone.

**Result: Balanced MSE removes 19 % of the stabilizing bias (0.58 → 0.47, CI excludes
zero) and improves nothing else** — every tail *ranking* metric is indistinguishable from
baseline while pooled r and MAE get significantly worse. LDS is dominated. **Neither
adopted as default.** With a frozen representation, loss reweighting moves predictions
but cannot create discrimination the features do not carry.

## Next steps

- [x] Full run + cluster bootstrap + figure — **done**.
- [x] Decision: **do not adopt** either loss as the default (see README).
- [ ] `details.md` + `build_report.py` → `report.pdf`.
- [ ] Revisit Balanced MSE **after** the backbone is unfrozen — that is the regime the
      published precedent (constraint-aware SPURS) actually covers.

## Blockers

None. Runs locally, CPU torch, ~1.5–2 h for the full 30 model fits.

## Log — newest first

### 2026-08-25 (later) — full run + bootstrap: fixes the bias, not the blindness

Full run (60 epochs, 2 seeds, batch 512), held-out Tsuboyama OOF:

| loss | r | ρ | MAE | stab_ρ | stab_bias | AUC | DetPr@30 | nDCG30 |
|---|---|---|---|---|---|---|---|---|
| mse | 0.767 | 0.767 | 0.466 | 0.263 | 0.584 | 0.851 | 0.567 | 0.359 |
| bmc | 0.706 | 0.771 | 0.553 | 0.270 | **0.471** | 0.855 | 0.633 | 0.383 |
| lds | 0.718 | 0.756 | 0.489 | 0.257 | 0.605 | 0.841 | 0.600 | 0.391 |

The raw table looked mildly encouraging, so significance was tested properly:
**cluster bootstrap over the 412 proteins** (400 resamples), paired difference vs MSE.

**BMC − MSE:** stab_bias **−0.113 [−0.144, −0.080]** (significant) · r
**−0.052 [−0.131, −0.001]** (significantly worse) · MAE **+0.088 [+0.076, +0.100]**
(significantly worse) · ρ +0.003, stab_ρ +0.005, AUC +0.003, DetPr@30 +0.017
(all CIs span zero).
**LDS − MSE:** ρ −0.011 and MAE +0.023 (both significantly worse); nothing better.

**Conclusion: Balanced MSE removes 19 % of the stabilizing bias and improves nothing
else.** With a frozen representation, reweighting moves predictions but cannot create
discrimination the features do not carry. The de-biasing is real but costs pooled r and
MAE, and the top-K detection curves are superimposed (figure panel 3) — it does not find
more stabilizing mutations. **Not adopted as default.**

Consistent with constraint-aware SPURS gaining from the same term while **fine-tuning**
its backbone, and with results/11 (S669 transfer showed no separation between losses,
r 0.37–0.39, tail ρ ≈ 0, because that error is domain shift).

*Caveat on the smoke test below:* its `mse` baseline showed r 0.790, the full run 0.767.
The 8-epoch models were under-trained rather than better; fold 2 is consistently the hard
fold (r 0.34–0.55 across runs). Single-run point estimates on this corpus are noisy at
the ±0.05 level — which is exactly why the bootstrap was necessary, and why the
smoke test's apparent BMC win on MAE/nDCG did not survive.

### 2026-08-25 — implemented and smoke-tested; full run launched

**Smoke test** (8 epochs, 1 seed — deliberately undertrained, directional only):

| loss | r | ρ | MAE | stab_ρ | stab_bias | auc_stab | detpr30 | ndcg30 |
|---|---|---|---|---|---|---|---|---|
| mse | 0.790 | 0.764 | 0.453 | 0.252 | **0.610** | 0.848 | 0.767 | 0.488 |
| bmc | 0.778 | **0.768** | 0.672 | **0.268** | **0.473** | **0.851** | 0.767 | **0.498** |
| lds | 0.558 | 0.705 | 0.664 | 0.218 | 0.510 | 0.819 | 0.633 | 0.425 |

- **BMC moves the target metric**: stabilizing bias 0.610 → 0.473 (−22 %), with
  stab_ρ, AUC and nDCG all marginally up and pooled ρ *slightly better* (0.764 → 0.768).
  Cost is overall MAE (0.453 → 0.672) — BMC optimises a density-corrected objective, so
  raw calibration on the bulk degrades.
- **LDS is dominated** on every metric; inverse-density weighting is too crude here.
- The `mse` baseline reproduces results/06 (r 0.790 vs 0.792) — the torch
  reimplementation is faithful.
- On S669 transfer nothing separated the losses at this epoch budget (r 0.41 for all
  three, stab_ρ ≈ 0 — S669 has only 69 stabilizing variants and, per results/11, its
  pooled error is dominated by domain shift, not the tail).

**Full run launched**: 60 epochs, 2 seeds, batch 512 →
`data/processed/_analysis/exp13.log`, `results.csv`.

**Implementation notes / gotchas**
- `sklearn.MLPRegressor` cannot express either loss (no `sample_weight`, no custom
  objective) — hence the torch rewrite. Architecture and augmentation kept identical to
  the project default so the comparison isolates the loss.
- BMC's `noise_var` must be **learned**, not fixed; with it fixed at 1.0 the loss is just
  a temperature-scaled softmax and the tail effect largely disappears.
- `ndcg30` gain is `max(0, −ΔΔG)`, ranking by *predicted* ΔΔG ascending (most
  stabilizing first), following Mutate Everything's stabilizing-ranking metrics.
