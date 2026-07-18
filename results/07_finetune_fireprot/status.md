# Status — 07_finetune_fireprot

**State:** 📋 Planned
**Last updated:** 2026-07-18

## Current state
New experiment (design locked, not yet implemented). **Question:** does *sequentially
fine-tuning* a Tsuboyama-pretrained raw-Δz ΔΔG regressor on FireProt improve accuracy
on FireProt — and does it degrade accuracy back on Tsuboyama (catastrophic forgetting)?
Builds on 05 (Tsuboyama→FireProt transfer, which showed error is governed by training
density in ΔΔG space): here we add FireProt labels to the *training* side and measure
the effect on both datasets, under homology-controlled splits.

**Scope note:** "fine-tune" = the downstream **regressor**, not Boltz. Raw-Δz features
are fixed Boltz-2 outputs; we adapt the sklearn MLP on top of them.

## Design (locked)

### Data
- **Train/test features (fixed, already extracted):**
  - Tsuboyama `tsuboyama_bench_fast/rawz_features.parquet` — 12,359 muts / **412 proteins**.
  - FireProt `fireprot_le200/features_summary.parquet` — 1,543 muts / **85 proteins**
    (≤200 aa corpus; chosen to start now — can extend to ≤500 later).
  - Both are the same 256 raw-Δz features (`zdiag_*` + `zpool_*`); ddg sign convention
    matches (positive = destabilizing).

### Splits — cross-dataset homology control (the crux)
Pool **all** WT sequences (412 Tsuboyama + 85 FireProt = 497) into one FASTA and cluster
with **MMseqs2** (`ddg.evaluation.cluster`). **Sweep identity thresholds 30 / 50 / 90 %.**
Assign **whole clusters** to train vs test (~80/20 by cluster) so that **no train/test
pair — within *or* across datasets — shares > threshold identity.** This catches the
subtle leak where a FireProt *test* protein is homologous to a Tsuboyama *training*
protein. Per threshold this yields four disjoint sets:
- **Tsu-train / Tsu-test** (Tsuboyama proteins in train / test clusters)
- **FP-finetune / FP-test** (FireProt proteins in train / test clusters)

Stratify the 80/20 cluster assignment so both datasets are represented in the test side.
(FireProt has only 85 proteins → ~17 FP-test proteins at 80/20; watch that FP-test stays
large enough to score. Consider a fixed seed + report FP-test protein count per threshold.)

### Conditions (same MLP everywhere; test each on BOTH held-out sets)
| Cond | Train | Purpose |
|---|---|---|
| **A. Tsuboyama-only** | Tsu-train | baseline (transfer with a Tsu holdout) |
| **D. Fine-tuned** | pretrain Tsu-train → warm-start continue on FP-finetune | the target |
| (B. FireProt-only) | FP-finetune | optional reference: FireProt alone |

Headline readouts, per identity threshold:
- **FireProt-test:** does D beat A? (benefit of adding FireProt labels)
- **Tsuboyama-test:** does D fall below A? (catastrophic forgetting)
- Report pooled r/ρ/RMSE/MAE + per-protein, and reuse the 05/06 diagnostics
  (`error_vs_ddg`, `density_vs_error`) — esp. whether fine-tuning fixes the FireProt
  out-of-range tail (the ΔΔG range FireProt covers but Tsuboyama does not).

### Fine-tune mechanism (sequential warm-start MLP)
- Base = the project MLP (5-seed ensemble of `MLPRegressor(256,128,64)`), `warm_start=True`.
- **Stage 1:** fit each member on Tsu-train to convergence.
- **Stage 2:** continue fit on FP-finetune with a **reduced `learning_rate_init`** (e.g.
  1e-4 vs 1e-3) and few epochs, warm-starting from the Stage-1 weights.
- **Avoid leakage when tuning finetune epochs/LR:** pick them via a small validation split
  carved from **FP-finetune** (or early-stopping on it), never FP-test.
- Open impl detail to verify: sklearn `warm_start` continues from `coefs_` but the Adam
  optimizer state is re-initialised per `fit`; confirm the second-stage `fit` behaves as
  intended (or use `partial_fit`). Prototype and sanity-check before the full sweep.

## Next steps (implementation checklist)
- [ ] Pull `fireprot_le200/wt_sequences.fasta` from the cluster (missing locally) — or
      rebuild from `mutations.csv`. Build a **combined** WT FASTA (Tsu + FP).
- [ ] Cluster the combined FASTA at 30/50/90 % with `ddg.evaluation.cluster`; write a
      combined `cluster_map_{30,50,90}.csv` (protein_id → cluster, spanning both datasets).
- [ ] Split module: whole-cluster 80/20 → {Tsu-train, Tsu-test, FP-finetune, FP-test},
      stratified, fixed seed; report per-set protein/mutation counts + leakage assertion
      (no shared cluster across train/test).
- [ ] New module `ddg/evaluation/finetune.py` (CLI like `transfer.py`): train A + D,
      predict both test sets, write per-condition × per-test-set metrics, scatter,
      `error_vs_ddg`, `density_vs_error`. Sweep the 3 thresholds.
- [ ] Sanity-check the warm-start finetune actually adapts (FP-test improves) without
      wrecking Tsu-test; validate finetune LR/epochs on a FP-finetune val split.
- [ ] Write README (What/Why/How + provenance) + `report.pdf` (paper-only; no
      provenance/plumbing — see `results/guidelines.md`); update `results/README.md` +
      `history.md`.

## Blockers
- None yet. Compute is light (sklearn on 256-dim features, ~40 s/MLP fit locally) — runs
  on this workstation, no cluster GPU needed (features already extracted).

## Log — newest first
### 2026-07-18 — planned + initialized
- Locked design with user: sequential warm-start fine-tune (not pooled); FireProt **≤200**
  to start; cross-dataset homology split **swept at 30/50/90 %**. Conditions A (Tsu-only)
  + D (fine-tuned), each tested on Tsu-test and FP-test. Created folder + this plan.
  Implementation not started.
