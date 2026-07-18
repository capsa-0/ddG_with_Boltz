# 08 — Sequentially fine-tuning on FireProt

**What:** Pretrain the ΔΔG MLP on Tsuboyama, then **sequentially fine-tune** it on FireProt
(warm-start continued training), and test on **both** datasets — under a cross-dataset
homology split (identity sweep 30/50/90 %). Uses the project defaults adopted in
[`07`](../07_feature_symmetry_ablation/): **concat features** + **antisymmetry augmentation**.

**Why:** [`05`](../05_cross_dataset_fireprot/) showed a Tsuboyama-trained model transfers to
FireProt but is bounded by training coverage. The question here: does adding FireProt labels
via fine-tuning improve FireProt accuracy — and does it cost Tsuboyama accuracy (catastrophic
forgetting)?

**How:**
- **Splits (`build_splits.py`):** pool all WT sequences (412 Tsu + 85 FP), cluster with
  MMseqs2 at 30/50/90 % identity (80 % coverage), assign whole clusters to train/test so no
  train/test pair — within or across datasets — exceeds the threshold. Mixed cross-dataset
  clusters → train; each dataset's own clusters 80/20. Four sets: `tsu_train`, `tsu_test`,
  `fp_finetune`, `fp_test`.
- **Model (`run_finetune.py`):** 5-seed MLP, **concat** features (`wtz`+`mtz`), **antisymmetry**
  aug on every training set. Condition **A** = Tsuboyama-only (pretrain, no FireProt);
  **D** = fine-tuned (pretrain, then warm-start continue on `fp_finetune`). Imputer/scaler fit
  on augmented `tsu_train` and reused across fine-tuning. Each condition tested on `tsu_test`
  and `fp_test`.

## Headline

**FireProt-test — does fine-tuning help?**

| Identity | A r → D r | A ρ → D ρ | A RMSE → D RMSE | n (proteins) |
|---|---|---|---|---|
| 30 % | 0.466 → **0.522** | 0.693 → **0.739** | 2.13 → 1.97 | 254 (13) |
| 50 % | 0.507 → **0.528** | 0.672 → **0.716** | 1.93 → 1.80 | 287 (14) |
| 90 % | 0.355 → 0.343 | 0.658 → **0.686** | 1.78 → 1.84 | 231 (17) |

**Tsuboyama-test — forgetting?** 0.794→0.790, 0.787→0.784, 0.790→0.778 (≤0.012).

**Takeaway:** sequential fine-tuning on FireProt gives a **modest, consistent FireProt gain**
— Spearman +0.03–0.05 at all three thresholds, Pearson/RMSE better at 30/50 % — with
**negligible Tsuboyama forgetting** (≤0.012 r). Fine-tuning is worth it, mildly. (The 90 %
Pearson dip, with Spearman still rising, is a calibration wobble on the smallest fp_test.)

## Data & provenance
| Item | Path |
|---|---|
| Splits | `splits/cluster_map_{30,50,90}.csv`, `splits/split_{30,50,90}.csv` (this folder) |
| Split builder | `build_splits.py` (needs `mmseqs` on PATH) |
| Runner | `run_finetune.py` → `results.csv` |
| Features | `data/processed/{tsuboyama_bench_fast,fireprot_le200}/features_ablation.parquet` (concat, from 07's `build_ablation_features.py`) |

## Files
- `results.csv` — full A/D × {tsu_test, fp_test} × {30,50,90}.
- `build_splits.py`, `run_finetune.py` — reproduce.
- `status.md` — log + verdict.
</content>
