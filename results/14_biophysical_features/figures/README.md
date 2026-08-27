# figures — 14_biophysical_features

Regenerate with `python results/14_biophysical_features/make_figures.py`.

Colour identifies the **evaluation set** in both figures (S669 orange, FireProt green);
every panel carries a legend or direct labels, so identity is never colour-alone. Palette
selected by running the colour-vision validator — see `details.md`.

All transfer numbers use the **no-augmentation** protocol so the two corpora are directly
comparable, and FireProt is **homology-filtered** to the 130 proteins sharing no
30 %-identity cluster with the training corpus.

### `01_what_generalizes.png` — what the readout is actually doing

- **A** — the ladder of readouts on both blind corpora, ordered from substitution identity
  to the diagonal. The project's current default (uniform pooling of levels) sits third
  from the bottom on both, while the diagonal alone leads at half the dimensionality.
- **B** — paired cluster-bootstrap differences, both corpora, `*` = CI excludes zero. The
  first three claims replicate; the fourth (contact weighting) does not.
- **C** — in-distribution skill against transfer skill. Red marks the configurations built
  on whole-chain pooling: they are the *best* in-distribution and the *worst* on transfer,
  which is why in-distribution holdout is a misleading selection signal here.

### `02_additions_that_failed.png` — the three biology-motivated additions

- **A** — contact weighting: significant on FireProt for rank correlation and error, but
  never for Pearson r, and exactly zero on S669.
- **B** — burial + biophysics: never exceeds the baseline; alone it reaches r = 0.014 on
  S669, and its raw chain-scale features degrade transfer.
- **C** — MSA conservation: every correlation and error interval contains zero against
  contact-weighted features, on a corpus with 100 % alignment coverage and median depth
  9,474, while stabilizing bias gets significantly worse.
