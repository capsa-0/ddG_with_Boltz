# Status — 12_error_anatomy

**State:** 🚧 In progress
**Last updated:** 2026-08-25

## Current state

Which mutations does the raw-Δz predictor actually get wrong? Run on two held-out
sets — the S669 blind benchmark (541 variants, regime D) and **12,359 out-of-fold
Tsuboyama predictions** (5-fold GroupKFold on `wt_id`, the results/06 protocol). All
error tables are computed on the **protein-centred** error as well as the raw error, so
class effects are not swamped by the per-protein calibration gap studied in results/11.

Burial is derived from **Boltz's own predicted distogram** (`pdrow` in the slim store,
64 bins over 2–22 Å → expected residues within 10 Å, |i−j|>2), so no external
structures or DSSP are needed. Resolved for 91 % of Tsuboyama and 100 % of S669.

Headline: **most apparent class effects are effect-size artifacts.** After normalising
MAE by the class's own spread of true ΔΔG, burial is flat and residue identity nearly
flat. Only three real weak spots survive: **→Pro**, **buried glycines**, and — by far
the largest — **stabilizing mutations**.

All numbers are in the Log below; the analysis completed and printed every table.

## Next steps

- [ ] Re-run `tsu_class_error.py` to regenerate `tsu_mut_classes.csv` and the Tsuboyama
      figure (the first run crashed on a matplotlib legend bug **after** printing all
      tables; script has since been patched to save the CSV before plotting).
      **Currently running.**
- [ ] Add `figures/README.md` index; add the Tsuboyama figure as `02_*`.
- [ ] `details.md` + `build_report.py` → `report.pdf`.
- [ ] **Follow-up experiment (the actionable one):** balanced-MSE loss reweighting for
      the stabilizing tail, following constraint-aware SPURS (arXiv 2606.08100), which
      bought S669 ρ 0.486 → 0.540 from loss changes alone. This is a *new* experiment,
      not part of this folder.

## Blockers

None. Runs locally in ~15–25 min.

## Log — newest first

### 2026-08-25 — error anatomy on S669 + 12,359 out-of-fold Tsuboyama

**Held-out Tsuboyama (OOF), overall:** r=0.777, ρ=0.784, MAE 0.44, n=12,359
(fold r: 0.825 / 0.688 / 0.801 / 0.803 / 0.796) — consistent with results/06's 0.792.
**FireProt-only model on all of Tsuboyama** (never saw any of it): r=0.604, ρ=0.654.

**Bias is ≈0 in every class on Tsuboyama** (−0.16 to +0.06 kcal/mol across all WT
residues, all mutant residues, all burial tertiles). Contrast S669, where per-class
bias ran to ±1.86 — that is the per-protein calibration failure (results/11), not a
mutation-class effect.

**Burial (Tsuboyama):** buried MAE 0.60 / mid 0.42 / exposed 0.35 — 1.7× worse buried,
same ratio as S669 (1.46 / 1.01 / 0.85). But sd(true) is 1.23 / 0.87 / 0.73, so
**MAE÷sd = 0.49 / 0.48 / 0.48 — flat**, and ρ is *highest* at buried sites
(0.79 vs 0.69 exposed). Burial does not degrade the model; it scales the errors.
This contradicts the common "worse on buried residues" claim for this representation.

**Gly/Pro (Tsuboyama), MAE ÷ sd(true):** →Pro **0.55** (n=446) · from-Gly 0.51 (782) ·
from-Pro 0.50 (330) · →Gly 0.46 (638) · other **0.44** (10,163). →Pro is the genuinely
worst class; from-Gly mildly worse.

**Burial × Gly:** buried from-Gly MAE 0.76, sd 1.18 → **0.64** (n=81) vs buried non-Gly
0.59/1.23 → 0.48 and exposed non-Gly 0.33/0.67 → 0.49. **Buried glycines are the worst
cell** — independently corroborating results/10, where FoldX blew up at buried glycines
in GLA; here Boltz's own error peaks at the same sites.

**Effect direction (Tsuboyama) — the real deficit:**
| direction | n | MAE | bias | ρ | sd(true) |
|---|---|---|---|---|---|
| stabilizing | 535 | 0.64 | **+0.56** | **0.27** | 0.30 |
| destabilizing | 6,120 | 0.54 | −0.21 | 0.71 | 0.89 |
| neutral | 5,704 | 0.32 | +0.13 | 0.36 | 0.25 |

The model systematically calls stabilizing mutations destabilizing, and its MAE is
**2× the class's own spread** — it cannot resolve them. This is the one weak spot that
is not an effect-size artifact, and it is the one that matters for protein engineering.

**Volume change (Tsuboyama):** smaller 0.47 / larger 0.44 / similar 0.41 — weak.

**S669 (regime D), for comparison:** overall r=0.462, ρ=0.451, MAE 1.11.
Burial: buried 1.46 / mid 1.01 / exposed 0.85 (sd 1.95 / 1.48 / 1.23).
Direction: stabilizing MAE 1.86, bias **+1.86**, ρ 0.11 (n=69); destabilizing 1.25,
bias −1.08, ρ 0.43; neutral 0.47. Worst WT residues C (1.85), W (1.78), N (1.59);
best K (0.68), V (0.78). Gly/Pro classes on S669 have too few variants to read
(from-Pro n=5, →Pro n=6).

**Prior art check.** Literature expects larger errors mutating away from Gly or to Pro,
and slightly worse on buried than exposed. We reproduce the Gly/Pro part and
**refute the burial part** once effect size is controlled for. results/10's GLA scan
had found glycines *better* there (per-mutation ρ +0.532 at G sites vs +0.445) — that
is consistent, since results/10 measured ranking (ρ), not magnitude, and ρ is indeed
not degraded at Gly sites here either.

**Gotcha:** the first run printed all tables then died in the figure legend
(`Line2D` has no `set_sizes`) *before* writing the CSV, losing ~20 min of MLP fits.
Script now writes `tsu_mut_classes.csv` immediately after prediction, before analysis.

**Code:** `tsu_class_error.py` (held-out Tsuboyama), `mut_class_error.py` (S669).
Intermediates land in `data/processed/_analysis/` (gitignored).
