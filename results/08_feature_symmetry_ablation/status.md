# Status — 08_feature_symmetry_ablation

**State:** 🚧 In progress — features built, ablation running
**Last updated:** 2026-07-18

## Current state
Ablation motivated by an **old notebook** (`notebooks/ddg_prediction.ipynb`, config
`experiment2.yaml` = `fireprot_full`) that reached **Pearson ~0.63 / Spearman ~0.65**
on a *within-FireProt* protein-holdout — higher than our current FireProt numbers. That
notebook differed from the current pipeline in three ways: **(1)** concat features, **(2)**
symmetry augmentation, **(3)** `mutate_across_msa`. This experiment isolates the first two
(the third needs a Boltz re-run and is out of scope here) and asks: **are they worth it,
on Tsuboyama and FireProt each?**

**Question:** within each dataset (protein-holdout), do either of these help?
- **Features:** `dz` = `zdiag`+`zpool` (current: the WT−mut **difference**) vs
  `concat` = `wtz`+`mtz` (old notebook: pooled **WT and mutant levels**, keeps both).
  Both 256-dim; note `zpool == mtz − wtz`, so concat is strictly more information.
- **Augmentation:** none vs **symmetry** — add the reversed mutation with negated ΔΔG to
  the TRAIN folds (ΔΔG(A→B) = −ΔΔG(B→A)). `dz`: negate features+ddg; `concat`: swap the
  two halves + negate ddg.

## Design
- **Corpora (fixed, already extracted):** Tsuboyama `tsuboyama_bench_fast` (12,359 muts /
  412 prot) and FireProt `fireprot_le200` (1,543 / 85). Features rebuilt from the slim
  stores by `build_ablation_features.py` → `features_ablation.parquet` (zdiag/zpool/wtz/mtz).
- **Eval:** `run_ablation.py` — GroupKFold(5) by `wt_id` (protein-holdout), OOF pooled
  Pearson/Spearman/RMSE/MAE. Model = project 5-seed MLP. 2 datasets × 2 features × 2 aug.
- **Scope note:** holds the MSA strategy fixed at whatever the corpora used (Tsuboyama;
  FireProt `mutate_first_row`). The notebook's `mutate_across_msa` is a *separate* variable
  (needs re-running Boltz) — not tested here.

## Next steps
- [ ] Read the results table; decide if concat and/or symmetry are worth adopting.
- [ ] If a win: fold into the pipeline (build_features + models) and note the retro effect
      on 05/07; else record the negative result.
- [ ] README + (if worthwhile) report.

## Blockers
- None. Local sklearn compute (features rebuilt from slim stores pulled from the cluster).

## Log — newest first
### 2026-07-18 — set up + running
- Reconciled the user's remembered ~0.65 FireProt: it was the old notebook's *within-FireProt*
  protein-holdout with concat features + symmetry aug + `mutate_across_msa` (not a transfer;
  our comparable transfer number is 05's 0.62). Built `build_ablation_features.py` (emits
  zdiag/zpool/wtz/mtz from the slim store) for both corpora; wrote `run_ablation.py` (2×2×2
  protein-holdout). FireProt features done (1,543×515); Tsuboyama building; ablation to run.
