# figures — 15_mave_stability_transfer

Regenerate with `python results/15_mave_stability_transfer/make_figures.py`.

| file | what it shows |
|---|---|
| `01_lopo_paired.png` | **The result.** (a) median Spearman ρ across the 13 MAVE datasets under leave-one-protein-out, per feature set, with the Rosetta and Boltz arms side by side; grey = the two models that carry no ΔΔG term at all, so they are arm-agnostic. (b) the paired difference Boltz − Rosetta with its 95 % bootstrap CI over the 11 proteins. Only ΔΔG-only clears zero. |
| `02_per_dataset_direct.png` | The same comparison without any model: direct \|Spearman ρ\| per dataset for Rosetta ΔΔG, our Boltz ΔΔG and GEMME ΔΔE. Sorted by Rosetta. The grey segment is the Rosetta→Boltz move; the right-hand column is Δ\|ρ\|. Our ΔΔG is ahead on 11 of 13 — the two exceptions are both UBI4 datasets, the one protein homologous to our training corpus. |

Colour: categorical slots 1–3 of the project's validated default palette
(blue / orange / aqua), assigned by identity in fixed order and checked with the
dataviz six-checks validator at `--pairs all` — worst all-pairs CVD ΔE 9.2, worst
normal-vision ΔE 24.0, both passing. Aqua's sub-3:1 contrast against the surface
raises a WARN, discharged as the rule requires: every value is directly labelled and
the source CSVs sit beside the figures.

Sign convention: ΔΔG **anti**-correlates with fitness (destabilizing → low fitness),
GEMME correlates positively. `01` plots signed ρ from the random forest (which predicts
fitness, so all arms are positive); `02` plots \|ρ\| so the three predictors are
comparable in magnitude, and says so on the axis.
