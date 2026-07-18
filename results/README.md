# results

One folder per result, each self-contained (`README.md` + `status.md` + `figures/`
+ `report`). Read a folder's `README.md` first; `status.md` is the living
progress log; `details.md` (where present) is the methods/provenance appendix.

**Adding or working on a result?** Follow [`guidelines.md`](guidelines.md) — what
each folder must contain, and the `status.md` logging rule (**append a log entry to
the folder's `status.md` every time you work on that experiment**, so half-finished
runs don't get lost).

**New here?** Read [`history.md`](history.md) — the narrative thread connecting the
experiments (why raw Δz, how generalization was proved, where it breaks).

| # | Result | Headline |
|---|--------|----------|
| [01_generalization](01_generalization/) | Generalization-holdout study of the raw-Δz ΔΔG predictor (Tsuboyama fast corpus, 12,359 mutations, HGB on 256 raw-Δz features). | Random-CV pooled **r = 0.78**; protein-holdout **0.70**; homology (30 % identity) **0.765**; per-protein mean r **0.81**. |
| [02_stress_extrapolation](02_stress_extrapolation/) | Extrapolation to the destabilizing tail — train on mild mutations (\|ΔΔG\|<1), test on the tail (ΔΔG>2). Wide corpus, 37,080 mutations. | Tail r **0.09**, fit slope **0.02**: the model interpolates but does **not** extrapolate beyond its training range. |
| [03_stress_learning_curve](03_stress_learning_curve/) | Pooled r vs. number of training proteins (proteins held out). Wide corpus, 37,080 mutations. | Near-saturated: **33 proteins → r 0.74**; 10× more (330) only adds **+0.05** (→ 0.79). |
| [04_no_msa_ablation](04_no_msa_ablation/) | MSA vs. single-sequence Boltz (`no_msa: true`), same corpus/features/model — isolates the evolutionary-signal contribution. | MSA is worth a uniform **~0.08–0.10 r** across all holdouts; structural prior alone still reaches **r 0.70**. |

### In progress
- **[05_cross_dataset_fireprot](05_cross_dataset_fireprot/)** — Tsuboyama-trained
  model tested on FireProt (1,543 muts / 82 proteins, ≤200 aa). Boltz features
  **done** on the cluster; **transfer evaluation not yet run** — see its
  [`status.md`](05_cross_dataset_fireprot/status.md).
