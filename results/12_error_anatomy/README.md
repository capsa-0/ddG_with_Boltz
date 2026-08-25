# 12 — Error anatomy: which mutations does the model get wrong?

**What:** A per-mutation-class breakdown of the raw-Δz predictor's error, on two
held-out sets: the **S669** blind benchmark (541 variants, regime D) and **12,359
out-of-fold Tsuboyama** predictions (5-fold GroupKFold on `wt_id`).

**Why:** results/01 and /06 hold out substitution type, source residue, target residue
and chemistry class — but those measure *generalization to unseen classes*, not *which
mutations are hard*. Nobody had asked the second question on a blind set. The literature
predicts two specific weak spots (Gly/Pro substitutions; buried residues), and both are
testable here.

**How:**
- Errors reported raw **and protein-centred** (per-protein mean error removed), so class
  effects are not confounded with the cross-protein calibration gap studied in
  results/11.
- Every class MAE is read against that class's own **sd of true ΔΔG** — without this,
  any class containing bigger effects looks "harder" purely by arithmetic.
- **Burial comes from Boltz's own predicted distogram**: the `pdrow` slice in the slim
  store (64 bins, 2–22 Å) gives P(d<10 Å) to every residue; summing over |i−j|>2 yields
  an expected contact number. No external structures or DSSP required. Resolved for
  91 % of Tsuboyama, 100 % of S669.

## Headline numbers

Held-out Tsuboyama: **r = 0.777, ρ = 0.784, MAE 0.44** (n = 12,359).
FireProt-only model on the same data (never saw any Tsuboyama): r = 0.604, ρ = 0.654.

**Most apparent class effects are effect-size artifacts.** Normalise and they vanish:

| Burial (Tsuboyama) | n | MAE | ρ | sd(true) | **MAE ÷ sd** |
|---|---|---|---|---|---|
| buried | 3,730 | 0.60 | 0.79 | 1.23 | **0.49** |
| mid | 3,729 | 0.42 | 0.74 | 0.87 | **0.48** |
| exposed | 3,730 | 0.35 | 0.69 | 0.73 | **0.48** |

Buried sites show 1.7× the raw MAE, but identical *relative* accuracy — and the best
ranking (ρ 0.79). **Burial does not degrade this model**, contradicting the usual claim.

Three weak spots do survive normalisation:

| Real weak spot | Evidence |
|---|---|
| **Stabilizing mutations** | bias **+0.56**, ρ **0.27**, MAE 0.64 against a class spread of only 0.30 — error is 2× the signal. The model calls stabilizing mutations destabilizing. |
| **→Pro** | MAE ÷ sd = **0.55** vs 0.44 for all other classes (n = 446). |
| **Buried glycines** | MAE ÷ sd = **0.64** (n = 81) vs 0.48 for buried non-Gly — independently corroborating results/10, where FoldX also broke down at buried glycines in GLA. |

**Conclusion:** the representation is uniformly accurate across burial and residue
identity once effect size is accounted for. Its one substantial deficit is the
**stabilizing tail** — precisely the regime that matters for protein engineering.

## Data & provenance

| Item | Path |
|---|---|
| Tsuboyama features | `data/processed/tsuboyama_bench_fast/features_ablation.parquet` |
| FireProt features | `data/processed/fireprot_le500/features_ablation.parquet` |
| S669 features | `data/processed/s669/features_ablation.parquet` |
| Burial source | `pdrow` in `data/processed/{tsuboyama_bench_fast,s669}/slim/*.npz` |
| Distogram binning | 64 bins, 2–22 Å (`external/boltz_modified/scripts/train/configs/structure.yaml`) |
| Model protocol | concat `wtz`+`mtz`, antisymmetry augmentation, 5-seed MLP `(256,128,64)` — same as `results/09_external_benchmarks/run_benchmarks.py` |
| Code | `tsu_class_error.py` (Tsuboyama OOF), `mut_class_error.py` (S669) |
| Intermediates | `data/processed/_analysis/` (gitignored) |

## Figures

- `figures/01_s669_mut_class_error.png` — S669: Gly/Pro classes, burial tertiles, and
  the amplitude-compression scatter.
- `figures/02_*` — Tsuboyama equivalent *(pending; regenerating)*.

## Next

The actionable follow-up is **balanced-MSE reweighting of the stabilizing tail**,
following constraint-aware SPURS (arXiv 2606.08100), which gained S669 ρ 0.486 → 0.540
from loss changes alone. Tracked as a separate experiment.
