# figures — 09_external_benchmarks

Regenerate both with `python results/09_external_benchmarks/make_figures.py`
(reads `results.csv`, `results_pre-correction.csv`, `ssym_antisymmetry.csv`).

| file | what it shows |
|---|---|
| `01_pooled_r_full_vs_filtered.png` | Pooled Pearson r per regime and benchmark, **full vs each regime's own 25 %-identity homology filter**. Regime A (Tsuboyama) has zero overlap with either benchmark, so its two bars are equal *by construction* — that is the control, not a result. The drop in B and D is the size of the leakage; it is largest on Ssym, where B's per-protein median falls 0.85 → 0.59. |
| `02_correction_and_antisymmetry.png` | **(a)** The estimator correction, per benchmark × subset × regime, with the before→after arrow annotated. The gain tracks training-set size (A +0.15/+0.16, B and D +0.02 to +0.06) rather than being uniform — the signature the epoch-count argument predicts. **(b)** The residual antisymmetry bias on Ssym's forward/reverse pairs, ± sd: only FireProt-only training is near-unbiased (+0.04); Tsuboyama-only over-destabilises (+0.25) and fine-tuning over-corrects (−0.23). |

**Read `01` with the caveat that the bars are not a like-for-like comparison across
regimes.** Each regime's "filtered" column drops a *different* protein set (its own
homologues), so only the `common25` subset in `results.csv` — one identical variant set for
all three — supports a cross-regime claim. That subset is what the report and the README
tables use for every comparative statement.

All numbers in both figures come from the corrected estimator. The pre-correction values are
kept in `results_pre-correction.csv` and appear only in panel (a) of `02`, explicitly labelled
as "before".
