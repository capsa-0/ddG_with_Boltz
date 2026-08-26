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
| [05_cross_dataset_fireprot](05_cross_dataset_fireprot/) | Tsuboyama-trained raw-Δz predictor tested — no refitting — on the independent **FireProt** dataset (3,205 muts / 138 proteins ≤500 aa, zero protein overlap). First test on a different dataset/assay. | Transfers: pooled **r 0.65 / ρ 0.66** (MLP; HGB 0.62), per-protein **median r 0.65** — on par with ThermoMPNN/AFToolkit. Under-predicts magnitude (slope 0.27). |
| [06_mlp_generalization](06_mlp_generalization/) | Experiment 01's holdout suite re-run with an **MLP** (5-seed ensemble) instead of HGB — same corpus/features/splits. Tests whether the result depends on the model or the representation. | MLP **matches/slightly beats** HGB on every holdout (random **0.80**, protein **0.79**, per-protein mean **0.83**): the generalization is a property of the **raw-Δz features**, not the tree model. |

| [07_feature_symmetry_ablation](07_feature_symmetry_ablation/) | Within-dataset (Tsuboyama & FireProt) 2×2 ablation of **concat vs Δz features** and **symmetry augmentation**, motivated by an old notebook that used both. | **Adopt concat+symmetry** (now the project default): concat ≥ Δz everywhere (free); symmetry helps FireProt (**+0.03 r**) but only with concat (on Δz it collapses Tsuboyama calibration). Neutral on Tsuboyama. |
| [08_finetune_fireprot](08_finetune_fireprot/) | Sequentially fine-tune the Tsuboyama-pretrained MLP on FireProt (concat + antisymmetry), test on **both** under a cross-dataset homology split (30/50/90 %). Does fine-tuning help FireProt without forgetting Tsuboyama? | **No (on ≤500):** fine-tuning does not reliably beat Tsuboyama-only transfer (A best in Pearson at 30/50 %; D only at 90 %) — the ≤200 gain washed out on the bigger test. FireProt-only still forgets Tsuboyama. Consistent with ThermoMPNN. |

| [09_external_benchmarks](09_external_benchmarks/) | Blind external benchmarks **S669** & **Ssym** under three training regimes, with MMseqs2 leakage control at 25/30 % identity. | S669 is the honest hard test: filtered pooled **r 0.40** (per-protein median **0.58–0.61**); Ssym's apparent edge was leakage. |
| [10_gla_scan](10_gla_scan/) | Full mutational scan of human α-galactosidase A (398 aa × 19 = 7,562 mutations), compared against FoldX. | First unlabelled real-target use; agreement concentrated where FoldX is in its buried-glycine clash regime. |
| [11_calibration_gap](11_calibration_gap/) | Is the missing cross-protein term a per-protein **offset**, and can it be predicted (from embeddings, descriptors, or ΔG(WT))? | **Closed as a direction.** Offset worth **+0.204 r** on S669 but only **+0.029** in-distribution; homologues share the mean ΔΔG (r **0.52**) but *not* the model's error on it (r 0.09) → it is **domain shift**, not a protein property. |
| [12_error_anatomy](12_error_anatomy/) | Which mutations are hard? Class breakdown on S669 + **12,359 out-of-fold** Tsuboyama, with burial taken from Boltz's own distogram. | Burial effects are **effect-size artifacts** (MAE÷sd flat at 0.48–0.49). Real deficits: **stabilizing mutations** (bias +0.56, ρ 0.27), →Pro, buried Gly. |

| [13_balanced_loss](13_balanced_loss/) | Does Balanced-MSE / LDS reweighting fix the stabilizing-tail deficit found in 12? Cluster-bootstrapped over 412 proteins. | **No.** BMC removes **19 %** of the stabilizing bias (0.58 → 0.47) but every tail *ranking* metric is unchanged, while r (−0.052) and MAE (+0.088) get worse. Loss reweighting cannot create discrimination a **frozen** representation lacks. |

| [15_mave_stability_transfer](15_mave_stability_transfer/) | 🚧 **In progress.** Does our ΔΔG predict **MAVE functional fitness** as well as Rosetta's? Reproduces Høie et al. 2022 (RF4Mave) with our ΔΔG swapped in. 11 proteins ≤200 aa / 13 MAVE datasets / 25,224 Boltz structures. | Harness reproduces their published LOPO baselines within **±0.011** (ΔΔG-only 0.249 vs 0.25, ΔΔE-only 0.409 vs 0.42, both 0.466 vs 0.47). Main comparison pending the Boltz run. |

### Archive
- **`old/`** — pre-refactor exploratory artifacts kept for reference (e.g. the
  P03050 Ala-scanning embedding gallery). Not part of the numbered result series.
