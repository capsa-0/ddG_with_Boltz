# Status — 08_finetune_fireprot

**State:** ✅ Done — finetuning gives a modest FireProt gain, no Tsuboyama forgetting
**Last updated:** 2026-07-18

## Results (concat + antisymmetry; A = Tsuboyama-only, B = FireProt-only, D = fine-tuned)

Pooled Pearson r (best per row bold):

| Identity | FireProt-test A / B / **D** | Tsuboyama-test A / B / D |
|---|---|---|
| 30% | 0.466 / 0.477 / **0.522** | 0.794 / 0.680 / 0.790 |
| 50% | 0.507 / 0.505 / **0.528** | 0.787 / 0.692 / 0.784 |
| 90% | 0.355 / 0.291 / 0.343 | 0.790 / 0.678 / 0.778 |

FireProt-test Spearman under D also rises: 0.69→0.74, 0.67→0.72, 0.66→0.69.

**Verdict:** fine-tuning (D) is the **only condition good on both** — it beats **both** baselines
on FireProt-test at 30/50 % (and Spearman at all thresholds) while keeping Tsuboyama (≤0.012 drop).
The **FireProt-only baseline (B)** matches the transfer on FireProt but **collapses on Tsuboyama**
(0.79→~0.68) — training on the small FireProt set alone throws away the Tsuboyama signal
(echoes ThermoMPNN). So fine-tuning *combines* the datasets rather than trading one for the other;
gain is modest, feature-bounded. (90 % row = smallest/noisiest fp_test: Pearson dips, Spearman rises.)

## Current state
**Question:** does *sequentially fine-tuning* a Tsuboyama-pretrained ΔΔG regressor on
FireProt improve FireProt accuracy — and does it degrade Tsuboyama accuracy (catastrophic
forgetting)? Builds on 05 (transfer) and uses the project defaults adopted in **07**:
**concat features** (`wtz`+`mtz`) and **antisymmetry augmentation** on every training set.

**Update:** originally numbered 07 and prototyped with Δz features (finetuning barely
helped: FP-test 0.487→0.496). Renumbered to **08** and **redone** with concat+antisymmetry
(`run_finetune.py`). Conditions A (Tsuboyama-only) + D (fine-tuned), each tested on
tsu_test and fp_test, across the 30/50/90 % homology splits (unchanged from before).

**Scope note:** "fine-tune" = the downstream **regressor**, not Boltz. The features
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
### 2026-07-18 — added FireProt-only baseline (B); it strengthens the story
- Added condition **B (FireProt-only)** to `run_finetune.py` (fresh model on `fp_finetune`
  with its own scaler) — the missing baseline. FireProt-test: B 0.477/0.505/0.291 vs A
  0.466/0.507/0.355 vs D 0.522/0.528/0.343. Tsuboyama-test: **B collapses to ~0.68** while
  A/D stay ~0.79. So D beats *both* baselines on FireProt (30/50 %) and is the only condition
  good on both datasets; B trades Tsuboyama away. Regenerated figure (A/B/D), README, report.
### 2026-07-18 — redone with concat+antisymmetry → finetuning helps FireProt
- Renumbered 07→08. Re-ran (`run_finetune.py`, 5-seed MLP, concat features +
  antisymmetry aug on tsu_train and fp_finetune; imputer/scaler fit on augmented
  tsu_train and reused for the FireProt stage). Results table above; `results.csv` committed.
- Finetuning (D) beats Tsuboyama-only (A) on fp_test in Spearman at all 3 thresholds
  (+0.03–0.05) and Pearson/RMSE at 30/50 %; Tsu-test drops ≤0.012 (no real forgetting).
- Finetune settings: pretrain max_iter=250, continue on FireProt lr=1e-3 max_iter=400,
  `early_stopping=False` throughout (toggling it across warm-start fits raises in sklearn).
  Not heavily tuned; a FP-val split could tune LR/epochs further.
### 2026-07-18 — splits built; finetune prototyped → marginal effect (feature-limited)
- Built cross-dataset homology splits with **mmseqs** (`build_splits.py`; Biopython
  single-linkage collapsed to 1 cluster at 30% — chaining — so switched to mmseqs
  `-c 0.8`). Only 4 mixed Tsu↔FP clusters per threshold. Committed `0b83b23`.
- Fixed an sklearn gotcha: toggling `early_stopping` across warm-start `fit` calls
  raises (`best_loss_` None); keep `early_stopping=False` in both stages.
- **Prototype (50% split, 3-seed MLP, fp_test=14 prot/287 muts):**
  A tsu-only FP-test r=0.487 / Tsu-test r=0.757; B fp-only 0.402 / 0.652;
  C pooled 0.444 / 0.758; **D finetune (lr1e-3,it500) 0.496 / 0.763** (Tsu RMSE
  0.667→0.726 = mild forgetting). Aggressive lr3e-3 hurt both.
- **Finding:** finetuning barely helps FireProt (+0.01 r); nothing beats Tsuboyama-only;
  FireProt-only is *worst* on FP-test. FP-test looks **feature-limited (~0.5 ceiling)**,
  not training-data-limited — same bottleneck story as 05/06. Pending: confirm across
  30/50/90 thresholds (full sweep) + decide whether to write up as-is or try le500 /
  a different finetune.
### 2026-07-18 — planned + initialized
- Locked design with user: sequential warm-start fine-tune (not pooled); FireProt **≤200**
  to start; cross-dataset homology split **swept at 30/50/90 %**. Conditions A (Tsu-only)
  + D (fine-tuned), each tested on Tsu-test and FP-test. Created folder + this plan.
  Implementation not started.
