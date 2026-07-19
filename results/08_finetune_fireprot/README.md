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
  aug on every training set. **A** = Tsuboyama-only (pretrain, no FireProt); **B** = FireProt-only
  (a fresh model on `fp_finetune` alone, its own scaler); **D** = fine-tuned (pretrain on
  Tsuboyama, then warm-start continue on `fp_finetune`, reusing the Tsuboyama scaler). Each
  condition tested on `tsu_test` and `fp_test`.

## Headline (pooled Pearson r; best per row bold)

| Identity | \| | FireProt-test A / B / **D** | \| | Tsuboyama-test A / B / D |
|---|---|---|---|---|
| 30 % | | 0.466 / 0.477 / **0.522** | | 0.794 / 0.680 / 0.790 |
| 50 % | | 0.507 / 0.505 / **0.528** | | 0.787 / 0.692 / 0.784 |
| 90 % | | 0.355 / 0.291 / 0.343 | | 0.790 / 0.678 / 0.778 |

(FireProt-test Spearman also improves under D: 0.69→**0.74**, 0.67→**0.72**, 0.66→**0.69**.
fp_test is 254/287/231 muts = 13/14/17 proteins.)

**Takeaway:** the fine-tuned model (**D**) is the **only condition that is good on both** — it
**beats both baselines on FireProt-test** at 30/50 % (and Spearman at all three thresholds),
*and* keeps Tsuboyama (≤0.012 drop). The **FireProt-only baseline (B)** is competitive with
Tsuboyama-transfer *on FireProt* but **collapses on Tsuboyama** (0.79→~0.68) — training on the
small FireProt set alone throws away the Tsuboyama signal (echoing ThermoMPNN, where FireProt-only
training degrades sharply). So fine-tuning genuinely *combines* the two datasets rather than
trading one for the other; the gain is modest but real. (The 90 % row is the smallest/noisiest
fp_test — Pearson dips while Spearman still rises.)

## Data & provenance
| Item | Path |
|---|---|
| Splits | `splits/cluster_map_{30,50,90}.csv`, `splits/split_{30,50,90}.csv` (this folder) |
| Split builder | `build_splits.py` (needs `mmseqs` on PATH) |
| Runner | `run_finetune.py` → `results.csv` |
| Features | `data/processed/{tsuboyama_bench_fast,fireprot_le200}/features_ablation.parquet` (concat, from 07's `build_ablation_features.py`) |

## Files
- `report.pdf` — paper-facing write-up (regenerate: `python results/08_finetune_fireprot/build_report.py`).
- `results.csv` — full A/B/D × {tsu_test, fp_test} × {30,50,90}.
- `build_splits.py`, `run_finetune.py` — reproduce.
- `status.md` — log + verdict.
</content>
