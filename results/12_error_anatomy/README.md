# 12 — Error anatomy: which mutations does the model get wrong?

**What:** A per-mutation-class breakdown of the predictor's error on held-out data, in
two passes:

1. **The original pass** — **S669** (541 variants, regime D) and **12,359 out-of-fold
   Tsuboyama** predictions (5-fold GroupKFold on `wt_id`), using the concat `wtz|mtz`
   readout that was the project default at the time.
2. **The transfer pass (2026-08-27)** — redone on **both** blind corpora, S669 and
   **FireProt ≤500 homology-filtered (3,102 variants / 130 proteins)**, with the readout
   results/14 adopted for transfer (the pair-track **diagonal alone**). The original pass
   used what results/14 later measured as the *worst* transfer readout, and S669 alone has
   no power for class cells (`from-Pro` n=5, `→Pro` n=6); FireProt had never been broken
   down by class at all. **Where the two passes disagree, the transfer pass is primary for
   any cross-corpus statement.**

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
| **→Pro** | MAE ÷ sd = **0.56** vs 0.43 for all other classes (n = 446). |
| **Buried glycines** | MAE ÷ sd = **0.64** (n = 81) vs 0.48 for buried non-Gly — independently corroborating results/10, where FoldX also broke down at buried glycines in GLA. |

## The transfer pass — what survives on the blind corpora

Redone with the transfer readout (pair-track diagonal) on both blind corpora. Reference:
S669 MAE 0.977, MAE ÷ sd 0.601, r 0.561 · FireProt-filtered MAE 0.811, MAE ÷ sd 0.508,
r 0.645. Contrasts are class-vs-all-other-mutations, cluster-bootstrapped over the 130
FireProt proteins.

| class | FireProt n | Δ(MAE ÷ sd) | 95 % CI | S669 n | Δ(MAE ÷ sd) | 95 % CI |
|---|---|---|---|---|---|---|
| **→Pro** | 80 | **+0.099** | [+0.049, +0.740] ✔ | 6 | −0.012 | [−0.042, +0.933] — no power |
| **from Pro** | 82 | **+0.098** | [+0.010, +0.216] ✔ | 6 | **+0.248** | [+0.171, +0.537] ✔ |
| **from Gly** | 157 | **+0.085** | [+0.007, +0.245] ✔ | 12 | +0.025 | [−0.240, +1.163] — no power |
| **→Gly** | 274 | −0.058 | [−0.116, +0.013] ✗ | 55 | −0.001 | [−0.141, +0.172] ✗ |
| from aromatic (FWY) | 352 | +0.016 | [−0.043, +0.104] ✗ | 64 | +0.119 | [+0.001, +0.299] ✔ |
| X→Ala | 723 | +0.004 | [−0.042, +0.045] ✗ | 220 | −0.101 | [−0.183, −0.007] ✔ |
| **near-isosteric (\|ΔVol\| < 30 Å³)** | 1,529 | **+0.070** | [+0.028, +0.113] ✔ | 252 | +0.065 | [−0.007, +0.138] ✗ |

The Gly/Pro deficit matches the in-distribution ordering above, and near-isosteric agrees
in sign on both corpora. **A useful refinement: `→Gly` is not a weak class.** What is hard
is *leaving* Gly, and Pro in either direction. Near-isosteric substitutions also survive
the floor control: they degrade in **ranking** too (ρ 0.589 vs 0.709 on FireProt; 0.438 vs
0.638 on S669), not just in MAE ÷ sd. *Caveat:* both metrics share one confounder — with
small true spread, experimental noise is a larger share of the variance, so "the model
cannot resolve small effects" and "the labels cannot either" are not separated here.

**Read the two columns together, not the FireProt one alone.** S669 is underpowered for
most cells (`→Pro` n = 6, `from Gly` n = 12) and it *disagrees* on the two classes where
it does have n: X→Ala comes out significantly **easier** there (−0.101) and neutral on
FireProt, and from-aromatic significant on S669 and neutral on FireProt. Only `from Pro`
clears zero on both. This is the same lesson as the retracted per-residue ranking below —
541 variants over 62 proteins will not settle a class-level question, and FireProt is
primary here because it has 130 proteins, not because it agrees with the prior.

**Correction — the per-residue error ranking does not replicate.** Spearman between the
two blind corpora for MAE ÷ sd by source residue is **+0.05 (p = 0.84, k = 17)**. Any
per-residue "worst/best" list read off S669 alone is noise from 541 variants and should
not be quoted; the earlier draft of this folder carried one. What *does* replicate across
corpora is the within-class ρ (+0.67, p = 0.05).

**What has structure instead: where the embedding beats an amino-acid lookup.** Scoring
`skill = 1 − MAE(diagonal) ÷ MAE(one-hot substitution)` against results/14's
substitution-identity control gives +0.224 [+0.167, +0.255] on FireProt, +0.145 on S669.
The extremes agree on both corpora: from **Q** (−0.29 S669 / −0.03 FireProt) and **W**
(+0.02 / +0.04) the embedding adds essentially nothing over a substitution matrix, while
from V (+0.28 / +0.27), L (+0.25 / +0.28) and A (+0.11 / +0.34) it earns its keep. By
pair: Y→F −0.03 (n = 64), W→A −0.01 (n = 14), W→F +0.02 (n = 24), K→R +0.03 (n = 15)
against L→A +0.35 / +0.37 (n = 58/30), I→A +0.20 / +0.40, V→A +0.22 / +0.29, G→A +0.25 /
+0.25. The full ranking replicates only at +0.40 (p = 0.11), so **the extremes are
defensible, the ordering is not.** Reading: the embedding's value concentrates on
hydrophobic core packing and degenerates toward a substitution matrix when the mutation
preserves the chemistry.

Note that **no class has negative skill with a CI excluding the corpus mean** — the
Gly/Pro deficit is not a collapse of the representation (skill there is a normal +0.21),
it is a magnitude problem.

**Conclusion:** the representation is uniformly accurate across burial and residue
identity once effect size is accounted for. Its substantial deficits are the
**stabilizing tail** — precisely the regime that matters for protein engineering — and,
on transfer, small-amplitude substitutions (Pro, leaving Gly, near-isosteric changes).
Separately, its *advantage over a plain amino-acid lookup* is not uniform: it is largest
for core-packing substitutions and near zero for chemistry-preserving ones.

## Data & provenance

| Item | Path |
|---|---|
| Tsuboyama features | `data/processed/tsuboyama_bench_fast/features_ablation.parquet` |
| FireProt features | `data/processed/fireprot_le500/features_ablation.parquet` |
| S669 features | `data/processed/s669/features_ablation.parquet` |
| Burial source | `pdrow` in `data/processed/{tsuboyama_bench_fast,s669}/slim/*.npz` |
| Distogram binning | 64 bins, 2–22 Å (`external/boltz_modified/scripts/train/configs/structure.yaml`) |
| Model protocol (original pass) | concat `wtz`+`mtz`, antisymmetry augmentation, 5-seed MLP `(256,128,64)` — same as `results/09_external_benchmarks/run_benchmarks.py` |
| Model protocol (transfer pass) | pair-track diagonal alone (`ddg.evaluation.labels.TRANSFER_BLOCKS`), no augmentation — the readout adopted in results/14 |
| Transfer-pass inputs | `data/processed/_analysis/exp14_{s669,fpfilt}_results_{locality*,onehot*}.csv` — the per-variant dumps results/14 wrote; nothing is re-trained and no GPU is used |
| Code | `tsu_class_error.py` (Tsuboyama OOF), `mut_class_error.py` (S669), `transfer_class_error.py` (both blind corpora, transfer readout), `indist_class_tables.py` (condenses the OOF per-variant table), `build_report.py` |
| Result tables | `indist_class_tables.csv` (held-out class summary), `transfer_class_tables.csv` (every class × corpus × grouping), `transfer_class_bootstrap.csv` (cluster-bootstrap contrasts), `transfer_replication.csv` (cross-corpus rank correlations) |
| Report | `report.pdf`, regenerated by `build_report.py` — every number is read from the committed tables above, so the PDF cannot drift from them |
| Intermediates | `data/processed/_analysis/` (gitignored) |

**Data gotcha:** S669 holds **17 repeated (protein, mutation) keys with different measured
ΔΔG** — genuine repeat measurements. Merging per-variant tables on that key
cross-products them and silently inflates 541 → 575 rows; the dumps are row-aligned, so
join by position.

## Figures

- `figures/01_s669_mut_class_error.png` — S669: Gly/Pro classes, burial tertiles, and
  the amplitude-compression scatter.
- `figures/02_tsuboyama_mut_class_error.png` — the same three panels on 12,359
  out-of-fold Tsuboyama predictions.
- `figures/03_transfer_class_error.png` — the transfer pass: class contrasts with
  cluster-bootstrap CIs, and skill over the amino-acid lookup by source residue and by
  substitution pair, on both blind corpora.

## Next

The actionable follow-up is **balanced-MSE reweighting of the stabilizing tail**,
following constraint-aware SPURS (arXiv 2606.08100), which gained S669 ρ 0.486 → 0.540
from loss changes alone. Tracked as a separate experiment (results/13, which found the
tail unmoved by loss reweighting alone).

From the transfer pass, one further question is open: the skill-over-lookup split says
the embedding contributes least exactly where the substitution preserves the chemistry
(Y→F, W→F, K→R) — i.e. where ΔΔG is decided by packing detail rather than by residue
type. Whether that is a limit of the *readout* (the diagonal is one 128-d vector at the
mutated site) or of the trunk itself is not answerable from these dumps; it would need a
readout that sees the local environment, not just the site.
