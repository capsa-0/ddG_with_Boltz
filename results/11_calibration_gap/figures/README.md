# figures — 11_calibration_gap

`01` is written by `offset_learn.py`; `02` by `make_figures.py`, which reads only the
committed tables (`split_half.csv`, `split_half_tsuboyama.csv`, `homology_share.csv`).

| file | what it shows |
|---|---|
| `01_per_protein_error.png` | Per-protein signed error and MAE by training regime on S669, and — panel 3 — per-protein error against the protein's **true** mean ΔΔG: slope 0.77, r 0.88. That near-unit slope is the finding: the model predicts close to the same mean for every protein, so its error tracks whatever that protein's mean actually was. The offset sd is 1.47 / 1.41 / 1.46 kcal/mol for regimes A / B / D. |
| `02_ceiling_and_sharing.png` | **The panel that carries the conclusion.** (a) The same oracle per-protein offset applied cross-dataset (S669, 0.511 → 0.655) and in-distribution (Tsuboyama, 0.779 → 0.808) — worth ~5× more across corpora than within one, so the model is already calibrated when provenance is shared. (b) Constructs of the same base structure share the protein's mean ΔΔG (r +0.516 ± 0.080) but **not** the model's error on it (r +0.090 ± 0.242). A fold property is shared; corpus context is not. |

**Both figures are on the corrected estimator.** The scripts behind them previously used
`max_iter=250, early_stopping=False` — the defect found in results/09, where `max_iter`
counts epochs so the regime with the most data over-trains hardest. Numbers from before
2026-08-27 do not match these.

Panel (b) of `02` is the whole argument in one comparison, and the error bars matter: with
39–51 groups the intervals are wide, so the claim rests on the **gap** between the two
correlations, not on either value's precision.
