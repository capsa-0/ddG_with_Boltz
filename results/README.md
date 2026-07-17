# results

One folder per result, each self-contained (`README.md` + `figures/` +
`report`). Read a folder's `README.md` first; `details.md` (where present) is the
methods/provenance appendix.

| # | Result | Headline |
|---|--------|----------|
| [01_generalization](01_generalization/) | Generalization-holdout study of the raw-Δz ΔΔG predictor (Tsuboyama fast corpus, 12,359 mutations, HGB on 256 raw-Δz features). | Random-CV pooled **r = 0.78**; protein-holdout **0.70**; homology (30 % identity) **0.765**; per-protein mean r **0.81**. |

### Planned / in progress
- **No-MSA vs MSA** — same corpus and features, Boltz run in single-sequence mode
  (`no_msa: true`), to isolate the evolutionary-signal contribution. (running)
- **02_stress_extrapolation** — train on mild mutations, test on the destabilizing
  tail (regression-to-the-mean weakness). See `docs/TODO.md` §3.
- **03_stress_learning_curve** — pooled r vs. number of training proteins.
- **04_cross_dataset_fireprot** — Tsuboyama-trained model tested on FireProt.
