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
- **Splits (`build_splits.py`):** pool all WT sequences (412 Tsu + 138 FP ≤500), cluster with
  MMseqs2 at 30/50/90 % identity (80 % coverage), assign whole clusters to train/test so no
  train/test pair — within or across datasets — exceeds the threshold. Mixed cross-dataset
  clusters → train; each dataset's own clusters 80/20. Four sets: `tsu_train`, `tsu_test`,
  `fp_finetune`, `fp_test`.
- **Model (`run_finetune.py`):** 5-seed MLP, **concat** features (`wtz`+`mtz`), **antisymmetry**
  aug on every training set. **A** = Tsuboyama-only (pretrain, no FireProt); **B** = FireProt-only
  (a fresh model on `fp_finetune` alone, its own scaler); **D** = fine-tuned (pretrain on
  Tsuboyama, then warm-start continue on `fp_finetune`, reusing the Tsuboyama scaler). Each
  condition tested on `tsu_test` and `fp_test`.

## Headline — FireProt ≤500 (pooled Pearson r; best per row bold)

| Identity | FireProt-test A / B / D | Tsuboyama-test A / B / D | fp_test (prot) |
|---|---|---|---|
| 30 % | **0.606** / 0.510 / 0.598 | 0.801 / 0.721 / 0.795 | 1,547 (25) |
| 50 % | **0.599** / 0.557 / 0.562 | 0.777 / 0.655 / 0.760 | 433 (27) |
| 90 % | 0.514 / 0.521 / **0.560** | 0.607* / 0.721 / 0.503* | 368 (27) |

(FireProt-test Spearman: A→D 0.58→0.59, 0.56→0.57, 0.53→0.61 — D marginally ≥ A at all three.
\*the 90 % Tsuboyama-test row is a calibration outlier — Pearson 0.61 vs Spearman 0.78 for A —
so the 90 % forgetting comparison is unreliable.)

**Takeaway (revised on ≤500):** with the ~2× larger, less noisy FireProt test set,
**fine-tuning no longer reliably beats Tsuboyama-only transfer.** On FireProt-test, plain
Tsuboyama-only (**A**) is the best in Pearson at 30 % and 50 %; D only wins at 90 %; in
Spearman D is marginally above A at all three. So the ≤200 "fine-tuning helps modestly"
result was largely an artifact of the tiny (13–17-protein) test set — on 25–27 test proteins
it washes out. This is the **field-consistent** outcome (ThermoMPNN: fine-tuning on FireProt
doesn't reliably help, and training on FireProt alone degrades). The one robust effect
survives: the **FireProt-only baseline (B) forgets Tsuboyama** (tsu-test 0.72/0.66 vs A's
0.80/0.78 at 30/50 %). Net: the winning recipe is **big-corpus pretraining + transfer**, not
fine-tuning on the small target set — consistent with 05's transfer result and the density-limited
picture (accuracy is set by the features/training coverage, not by exposure to FireProt labels).

## Data & provenance
| Item | Path |
|---|---|
| Splits | `splits/cluster_map_{30,50,90}.csv`, `splits/split_{30,50,90}.csv` (this folder) |
| Split builder | `build_splits.py` (needs `mmseqs` on PATH) |
| Runner | `run_finetune.py` → `results.csv` |
| Features | `data/processed/{tsuboyama_bench_fast,fireprot_le500}/features_ablation.parquet` (concat, from 07's `build_ablation_features.py`) |

## Files
- `report.pdf` — paper-facing write-up (regenerate: `python results/08_finetune_fireprot/build_report.py`).
- `results.csv` — full A/B/D × {tsu_test, fp_test} × {30,50,90}.
- `build_splits.py`, `run_finetune.py` — reproduce.
- `status.md` — log + verdict.
</content>
