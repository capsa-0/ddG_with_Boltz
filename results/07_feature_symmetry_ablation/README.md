# 07 — Feature form & symmetry augmentation ablation

**What:** A 2×2 ablation, run **within each dataset** (Tsuboyama and FireProt separately,
protein-holdout), of two techniques an earlier notebook used but the current pipeline
dropped: **concat vs Δz features**, and **symmetry augmentation** on/off.

**Why:** An old notebook (`notebooks/ddg_prediction.ipynb`) reached Pearson ~0.63 on a
within-FireProt protein-holdout — higher than our current FireProt numbers. It differed in
three ways: concat features, symmetry augmentation, and `mutate_across_msa`. This isolates
the first two (the third needs a Boltz re-run) to decide whether they are worth adopting.

**How:**
- **Features** (256-dim each, from the slim store via `build_ablation_features.py`):
  `dz` = `zdiag`+`zpool` (current — the WT−mutant **difference**) vs
  `concat` = `wtz`+`mtz` (pooled **WT and mutant levels**; `zpool = mtz − wtz`).
- **Augmentation:** symmetry — add each reversed mutation with negated ΔΔG to the TRAIN
  folds (ΔΔG(A→B) = −ΔΔG(B→A)). `dz`: negate features + ddg; `concat`: swap the two halves.
- **Eval:** `run_ablation.py` — GroupKFold(5) by `wt_id`, OOF pooled metrics, 5-seed MLP.
- **Corpora:** `tsuboyama_bench_fast` (12,359 / 412) and `fireprot_le200` (1,543 / 85).

## Headline (pooled Pearson r, protein-holdout)

| Dataset | Feature | no-aug | +symmetry |
|---|---|---|---|
| Tsuboyama | Δz (current) | 0.792 | **0.598** |
| Tsuboyama | concat | 0.801 | 0.799 |
| FireProt | Δz (current) | 0.573 | 0.593 |
| FireProt | concat | 0.578 | **0.605** |

**Verdict — adopt `concat + symmetry`:**
- **concat ≥ Δz everywhere** — small, consistent, and free (strictly more information).
- **symmetry is representation-dependent.** On Δz it forces an odd-function model that
  fights Tsuboyama's destabilizing skew → Pearson 0.79→0.60 (but Spearman intact ≈ 0.78 →
  a *calibration* collapse, not a ranking one). On **concat** the reverse mutation is a
  natural input transform (swap halves), so it is safe on Tsuboyama and **lifts FireProt**.
- Best combo vs the current baseline (Δz, no-aug): **Tsuboyama 0.799 vs 0.792** (noise),
  **FireProt 0.605 vs 0.573 → +0.032 Pearson** (RMSE 1.42→1.38).
- The residual gap to the notebook's ~0.63 on FireProt is the two variables not tested
  here: **`mutate_across_msa`** and the larger **≤500** corpus.

## Data & provenance
| Item | Path |
|---|---|
| Feature builder | `build_ablation_features.py` (slim store → `features_ablation.parquet`) |
| Ablation | `run_ablation.py` → `results.csv` (this folder) |
| Tsuboyama features | `data/processed/tsuboyama_bench_fast/features_ablation.parquet` (12,359×515) |
| FireProt features | `data/processed/fireprot_le200/features_ablation.parquet` (1,543×515) |

`data/processed/` is gitignored; only the scripts + `results.csv` are committed.

## Files
- `report.pdf` — paper-facing write-up (regenerate: `python results/07_feature_symmetry_ablation/build_report.py`).
- `results.csv` — the 8-row table (dataset × feature × augment).
- `build_ablation_features.py`, `run_ablation.py` — reproduce from the slim stores.
- `status.md` — log + full verdict.
</content>
