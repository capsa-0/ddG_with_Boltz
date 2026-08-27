# figures — 15_mave_stability_transfer

Regenerate `01`/`02` with `make_figures.py`, `03`/`04` with `paper_figures.py`.

| file | what it shows |
|---|---|
| `01_lopo_paired.png` | **The result.** (a) median Spearman ρ across the 13 MAVE datasets under leave-one-protein-out, per feature set, with the Rosetta and Boltz arms side by side; grey = the two models that carry no ΔΔG term at all, so they are arm-agnostic. (b) the paired difference Boltz − Rosetta with its 95 % bootstrap CI over the 11 proteins. Only ΔΔG-only clears zero. |
| `02_per_dataset_direct.png` | The same comparison without any model: direct \|Spearman ρ\| per dataset for Rosetta ΔΔG, our Boltz ΔΔG and GEMME ΔΔE. Sorted by Rosetta. The grey segment is the Rosetta→Boltz move; the right-hand column is Δ\|ρ\|. Our ΔΔG is ahead on 11 of 13 — the two exceptions are both UBI4 datasets, the one protein homologous to our training corpus. |
| `03_landscape_reproduction.png` | **Their Figure 1, with our ΔΔG in Rosetta's place.** Top: the stability–conservation landscape (ΔΔG × GEMME ΔΔE, coloured by fitness), both arms. Bottom: their Fig 1B sector grid — % loss-of-function per sector, Rosetta / ours / difference. The Rosetta arm reproduces the two corners the paper quotes (84 % / 96 % against their published 81 % / 93 %), which is what licenses reading the other two panels. Our arm is drawn at **quantile-matched** cuts because our ΔΔG is amplitude-compressed (sd 0.97 vs 2.14 kcal/mol) and the paper's absolute cuts would leave its top column nearly empty. Cells with n < 50 are greyed — the loudest number on the difference panel (+32) sits on n = 4. |
| `04_conservation_strata.png` | **Where the advantage over Rosetta comes from.** AUC for detecting loss-of-function, per conservation quartile and pooled. Pooled we beat Rosetta by +0.048 [+0.014, +0.079]; holding conservation fixed, by +0.021 [−0.010, +0.052] — more than half the advantage is explained by our ΔΔG carrying conservation signal, which is the MSA confound `../README.md` flags as the open question, bounded here with no GPU. The residual is positive in all four strata, so it is underpowered rather than absent. |

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

**ΔΔE orientation — read before reusing the label table.** The `gemme_dde` column of
`data/raw/mave_hoie_le200_labels.csv` is the PRISM `gemme_score_01` verbatim, and it runs
**opposite to the paper's ΔΔE**: high = evolutionarily tolerated, so it correlates
*positively* with fitness (pooled ρ +0.27). That is why `layer1_direct.csv` shows
`rho_gemme > 0` while both ΔΔG arms are negative. `03`/`04` plot ΔΔE = 1 − `gemme_dde`,
and `paper_figures.py` asserts that choice against the paper's published corner
percentages instead of trusting the column name.
