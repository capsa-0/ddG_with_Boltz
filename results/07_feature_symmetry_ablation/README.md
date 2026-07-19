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
- **Corpora:** `tsuboyama_bench_fast` (12,359 / 412) and `fireprot_le500` (**3,205 / 138**).

## Headline (pooled Pearson r, protein-holdout)

| Dataset | Feature | no-aug | +symmetry |
|---|---|---|---|
| Tsuboyama | Δz (current) | 0.792 | **0.598** |
| Tsuboyama | concat | 0.801 | 0.799 |
| FireProt ≤500 | Δz (current) | 0.575 | 0.579 |
| FireProt ≤500 | concat | 0.508 | **0.588** |

**Verdict — adopt `concat + symmetry`:**
- **`concat + symmetry` is the best FireProt config (0.588)** and neutral on Tsuboyama
  (0.799 vs the Δz-no-aug baseline 0.792) — the combo to use.
- **symmetry is representation-dependent.** On Δz it forces an odd-function model that
  fights Tsuboyama's destabilizing skew → Pearson 0.79→0.60 (Spearman intact ≈ 0.78 →
  a *calibration* collapse, not a ranking one). On **concat** the reverse mutation is a
  natural input transform (swap halves), so it is safe on Tsuboyama.
- **On the ≤500 corpus symmetry is what makes concat win:** concat *alone* underperforms
  Δz on FireProt (0.508 vs 0.575), but **concat + symmetry lifts it to the top (0.588)**,
  ahead of both Δz variants and with lower RMSE (1.29 vs 1.32). (On the smaller ≤200 set
  concat was already ahead without augmentation; on the harder ≤500 set augmentation is
  needed — but the "adopt concat + symmetry" conclusion is unchanged.)
- Residual gap to the old notebook's ~0.63 FireProt: the one variable not tested here,
  **`mutate_across_msa`** (needs a fresh Boltz run).

## Data & provenance
| Item | Path |
|---|---|
| Feature builder | `build_ablation_features.py` (slim store → `features_ablation.parquet`) |
| Ablation | `run_ablation.py` → `results.csv` (this folder) |
| Tsuboyama features | `data/processed/tsuboyama_bench_fast/features_ablation.parquet` (12,359×515) |
| FireProt features | `data/processed/fireprot_le500/features_ablation.parquet` (3,205×515) |

`data/processed/` is gitignored; only the scripts + `results.csv` are committed.

## Files
- `report.pdf` — paper-facing write-up (regenerate: `python results/07_feature_symmetry_ablation/build_report.py`).
- `results.csv` — the 8-row table (dataset × feature × augment).
- `build_ablation_features.py`, `run_ablation.py` — reproduce from the slim stores.
- `status.md` — log + full verdict.
</content>
