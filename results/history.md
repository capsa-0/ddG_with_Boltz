# Project history — the path through the experiments

The narrative thread linking the results in this folder: how we got from "extract
something from Boltz-2" to a specific, validated ΔΔG predictor. Read this first for
the *why*; each `NN_*/` folder has the *what* and the numbers.

---

## 0. The goal

Predict the change in folding free energy (**ΔΔG**) of a single-point mutation
**without running full structure prediction**. Boltz-2 is run in an
**embeddings-only** mode: we keep the trunk representations for the wild-type and
the mutant sequence, and regress ΔΔG on features derived from their difference.
The open question was never "does a structure model know about stability" but
**which internal representation carries that signal, and in what form.**

The trunk exposes three tensors per structure:
- `s` — single track, one vector per residue (L × 384),
- `z` — pair track, residue×residue (L × L × 128),
- `pdistogram` — predicted distance distribution (L × L × bins).

## 1. Which embedding? (feature-representation experiment)

We compared feature representations built from the WT−mutant difference at the
mutated residue, all with the same model (HGB) and CV protocol. Two families:

- **Summary statistics** — reduce each difference slice (`s`, `z`, `pdistogram`) to
  ~14 scalar moments (mean, SD, gini, entropy, skew, …), concatenated to **653
  features**. This was the original pipeline.
- **Raw Δz** — keep the pair track *raw*: `Δz_diagonal` (128) = `mut_z[i,i]−wt_z[i,i]`
  plus `Δz_row-pooled` (128) = residue-mean of `mut_z[i,:]−wt_z[i,:]`. **256
  features**, no summarizing.

**Result (5-fold CV, fast corpus):**

| Representation | # feat | CV r |
|---|---|---|
| ALL summary statistics | 653 | 0.710 |
| raw Δs | 384 | 0.658 |
| **raw Δz (diagonal + pooled)** | **256** | **0.780** |
| raw Δs + raw Δz | 640 | 0.774 |

Two things fell out of this:
1. **Raw Δz beats the entire 653-feature summary set** (0.780 vs 0.710) at a third
   the width — summarizing `z` into moments throws away most of the signal.
2. **Adding `s` slightly *hurts*** (0.780 → 0.774): the single track is redundant
   with the raw pair track. So `s` is optional (kept only as a config switch).

→ **Decision: raw Δz is the representation.** (Figure: `01_generalization/figures/01_feature_comparison_raw_vs_summary.png`.)

## 2. Making it the pipeline (refactor)

That decision was baked into the code: the `features` step now emits raw Δz
directly from the slim embedding store (`ddg/features/build_features.py`), and the
old summary-statistics extractor (`ddg/exploration/`) was removed. `slim.keep_s`
became an opt-in switch (default off), since `s` is redundant.

## 3. Does it generalize? → `01_generalization/`

Random CV overstates real performance because mutations of the same protein leak
across folds. We ran a holdout suite of increasing strictness
(`experiment_configs/tsuboyama_bench_fast.yaml`, 12,359 mutations):

| Holdout | pooled r |
|---|---|
| Random | 0.783 |
| Protein (unseen proteins) | 0.702 |
| Homology (30 / 50 / 90 % identity) | 0.765 / 0.766 / 0.772 |
| De-novo (natural ↔ designed transfer) | 0.615 |

**It generalizes.** Holding out whole proteins costs ~0.08 r; homology clustering
barely adds to that; the hardest transfer (natural↔designed) still holds at 0.62.
Per-protein mean r = 0.81.

## 4. Where does it break? → `02_stress_extrapolation/`, `03_stress_learning_curve/`

Generalization across *proteins* is not the same as across *effect sizes* or *data
budgets*. Two stress tests on the wide corpus
(`data/processed/tsuboyama_bench_wide/`, 37,080 mutations, raw Δz):

- **Extrapolation to the destabilizing tail** (`02_`) — train only on mild
  mutations (|ΔΔG|<1), test on the strongly destabilizing tail (ΔΔG>2). The model
  **collapses**: tail r ≈ 0.09, fit slope ≈ 0.02, and predictions cap at the
  training range (never exceed ~0.9 kcal/mol while true values reach 5.7). This is
  the regression-to-the-mean weakness in its extreme form — the predictor
  interpolates, it does not extrapolate beyond the ΔΔG range it was trained on.
- **Learning curve** (`03_`) — pooled r vs. number of training proteins, proteins
  held out. **Near-saturated**: 33 proteins already give r ≈ 0.74, and a 10×
  increase to 330 proteins adds only ~0.05 (→ 0.79). The representation is strong
  enough that the model is not badly data-starved — more proteins help only
  marginally.

## 5. Does the evolutionary signal matter? → `04_no_msa_ablation/`

Boltz normally sees a multiple-sequence alignment (MSA) per protein. We re-ran the
fast corpus with Boltz in **single-sequence mode** (`no_msa: true`,
`experiment_configs/tsuboyama_bench_fast_nomsa.yaml`) — identical corpus, features,
and model, differing *only* in the MSA — to isolate how much of the ΔΔG signal
comes from the evolutionary input vs. the structural prior.

**The MSA is worth a uniform ~0.08–0.10 pooled r** across every holdout (mean Δr ≈
−0.086), largest for de-novo transfer (−0.099). But single-sequence Boltz still
reaches r = 0.70 (random) / 0.69 (unseen proteins): **most of the ΔΔG signal is
structural**, the MSA is a real but modest add-on, and the predictor degrades
gracefully without it (useful, since the MSA server is rate-limited/flaky).

## 6. Is it the representation or the model? → `06_mlp_generalization/`

Experiment 01 proved generalization using gradient-boosted trees (HGB). To check
whether that's a property of the **representation** or just of one estimator, we
re-ran the entire holdout suite unchanged (same fast corpus, same 256 raw-Δz
features, same splits) with a **neural-network MLP** instead — a 5-seed ensemble of
a deeper, L2-regularized MLP. (A single-seed MLP was too high-variance across group
folds — non-monotonic homology numbers — so it's averaged over seeds.)

The MLP **matches or slightly beats HGB on every holdout** (random 0.80 vs 0.78,
protein 0.79 vs 0.77, homology +~0.017, per-protein mean 0.83 vs 0.81; even on the
hard de-novo/substitution/chemistry transfers). **Two independent model families
land on the same generalization profile → the signal lives in the raw-Δz features,
not the regressor.** The regression-to-the-mean weakness persists, confirming it's a
property of the features/objective (cf. `02_`), not the model.

## 7. Does it survive a whole different dataset? → `05_cross_dataset_fireprot/`

Every holdout so far splits *within* Tsuboyama. The real test is transfer to an
independently curated dataset. We trained on **all 12,359 Tsuboyama mutations** and
predicted — with no refitting — **all 3,205 FireProt ≤500 aa mutations** (138 real
proteins, raw-Δz features). FireProt is a genuine adversary: **zero `wt_id` overlap**
with the training set, a different stability assay, and a much wider ΔΔG range
([−13.7, +12.0]).

It **transfers**: pooled **r = 0.65 / ρ = 0.66** (MLP; HGB 0.62, within noise —
same representation-not-model story as `06_`), per-protein **median r = 0.65**. This is
**on par with the published state of the art** — ThermoMPNN and the AFToolkit framework
report ~0.65 Pearson transferring to FireProt from the same Megascale/Tsuboyama data with
AlphaFold2 / GNN backbones; the Boltz-2 raw-Δz pipeline reaches the same level with a
simple regressor. The ceiling is again **magnitude, not ranking** (cf. `02_`): slope 0.27,
predictions span ~40 % of the true spread, and error is almost perfectly anti-correlated
with training density in ΔΔG space (ρ = −0.96). Useful for **ranking/triage**, not absolute
ΔΔG on out-of-range mutations.

## 8. Two improvements adopted as the default → `07_feature_symmetry_ablation/`

Revisiting an older notebook that reached ~0.63 on FireProt surfaced two techniques the
pipeline had dropped. A within-dataset 2×2 ablation (Tsuboyama and FireProt, protein
holdout) settled both, and **from here on they are the project default:**

- **Concat features instead of the Δz difference.** For the mutated residue we now keep
  the mutant residue's pooled pair-vector *and* the wild-type's — `concat = [wtz, mtz]`
  (256-dim) — instead of only their difference `zpool = mtz − wtz`. Strictly more
  information; concat ≥ Δz on both datasets (small but consistent).
- **Antisymmetry augmentation.** Each training mutation is paired with its reverse
  (ΔΔG(A→B) = −ΔΔG(B→A)); with concat this is a natural input transform (swap the two
  halves). It lifts FireProt (**+0.03 Pearson**) and is neutral on Tsuboyama. Crucially it
  is *only* safe with concat — applied to Δz (feature negation) it forces an odd-function
  model that collapses Tsuboyama's calibration (Pearson 0.79→0.60, Spearman intact).

Net: **concat + antisymmetry** — neutral on Tsuboyama, best FireProt config. The residual
gap to the old notebook's 0.63 is the one variable not yet tested (`mutate_across_msa`).
Later experiments use these defaults.

## 9. Does fine-tuning on FireProt help? → `08_finetune_fireprot/`

Given a Tsuboyama-pretrained model, can fine-tuning on FireProt improve FireProt accuracy
without forgetting Tsuboyama? Under a cross-dataset homology split (Tsuboyama + FireProt ≤500
clustered together, 30/50/90 % identity), we compared Tsuboyama-only (A), FireProt-only (B),
and fine-tuned (D). On the ≤500 test set (25–27 held-out proteins) **fine-tuning does not
reliably beat plain transfer** — Tsuboyama-only is the best FireProt-test model in Pearson at
30/50 %, fine-tuning only at 90 %, rank correlation ~tied. (A smaller ≤200 test had hinted at a
modest gain; it washed out on the larger set.) The one robust effect is that **FireProt-only
training forgets Tsuboyama.** This matches the field (ThermoMPNN) and this project's own
density-limited picture: accuracy is set by the features and the coverage of the large
pretraining corpus, not by exposure to the small target set. **Big-corpus pretraining +
transfer is the recipe.**

---

### Result folders
- `01_generalization/` — the holdout study (the "it works and generalizes" result).
- `02_stress_extrapolation/` — the destabilizing-tail extrapolation failure.
- `03_stress_learning_curve/` — data-efficiency curve.
- `04_no_msa_ablation/` — MSA vs. single-sequence Boltz (evolutionary-signal value).
- `05_cross_dataset_fireprot/` — transfer to an independent dataset (FireProt).
- `06_mlp_generalization/` — 01's suite with an MLP (representation vs model check).
- `07_feature_symmetry_ablation/` — concat features + antisymmetry adopted as default.
- `08_finetune_fireprot/` — sequentially fine-tune on FireProt (concat+antisymmetry).

See each folder's `README.md` for exact configs, data paths, and numbers.
