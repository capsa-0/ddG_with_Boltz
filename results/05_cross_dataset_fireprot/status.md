# Status — 05_cross_dataset_fireprot

**State:** ✅ Done — full FireProt corpus (1,543 muts / 85 proteins) extracted and the
Tsuboyama→FireProt transfer eval run. Headline: pooled **r=0.62 / ρ=0.68** (MLP),
per-protein median r=0.67. README/figures written.
**Last updated:** 2026-07-18

## Current state
Cross-dataset transfer test: does the **Tsuboyama-trained** raw-Δz ΔΔG predictor
generalize to the independent **FireProt** dataset? **Yes** — trained on all 12,359
Tsuboyama mutations, tested with no refitting on all 1,543 FireProt mutations (85
proteins, zero `wt_id` overlap): pooled **r=0.621, ρ=0.684, RMSE=1.41** (MLP; HGB
r=0.607). Per-protein **median r=0.67** (70 % of proteins r>0.5). Same magnitude
ceiling as 02/06: fit slope 0.26 — ranks well, under-predicts the tails.
See `README.md` for the full write-up.

**Correction (2026-07-18):** the earlier claim that the FireProt feature pipeline was
"complete" was wrong. `features_summary.parquet` (built 2026-07-17 15:39) contained
only **773 mutations / 54 proteins**, not 1,543 / 82 — the `slim` step ran on a
**partially finished** predict array (the manifest has no `predict` entry), and
`delete_raw: true` then deleted the raw NPZs, so the missing ~770 structures cannot
be cheaply resumed. The user wants the **full** FireProt corpus, so predict is being
re-run from scratch (all 1,597 query structures) → slim → features.

- **Config:** `experiment_configs/fireprot_le200.yaml` (dataset_type `fireprot`,
  `mutate_wt_msa` / `mutate_first_row`, max_msa 1000, `keep_s: false`, `delete_raw: true`).
- **Corpus:** FireProt ≤200 aa — **1,543 mutations / 82 proteins**, 0 dropped
  (`data/processed/fireprot_le200/dataset_report.json` all-clean).
- **Cluster processed dir:** `data/processed/fireprot_le200/` — full
  `features_summary.parquet` (1,543 × 256, rebuilt 2026-07-18 by jobs 212209–212212).
- **Transfer output:** `data/processed/fireprot_le200/transfer_from_tsuboyama{,_hgb}/`
  (pulled locally). Note: `python -m ddg.evaluation` does *within-dataset* holdouts —
  the cross-dataset transfer is a separate tool, `ddg/evaluation/transfer.py`.

## Next steps
- [x] Full FireProt corpus extracted (1,543 / 85) after the `pdb_id`-fallback fix.
- [x] Transfer eval run (`ddg/evaluation/transfer.py`, MLP + HGB); sign convention
      confirmed identical (no flip). README + figures written; indexes updated.
- [ ] Optional: re-run the transfer with the **wide** Tsuboyama corpus
      (`tsuboyama_bench_wide` `rawz_features.parquet`, 37,080 muts) as the training
      set to see whether more/broader training data lifts the FireProt transfer.
- [ ] Optional: commit `data/raw/fireprot_le200.csv` + `experiment_configs/fireprot_le200.yaml`
      (currently untracked on both machines) so the experiment is fully reproducible
      from git alone.

## Blockers
- None active. **Historical:** the predict step repeatedly failed at startup when
  shards landed on bad GPU nodes `nodo3` / `nodo5` (SIGSEGV / ld.so crash — see
  CLAUDE.md "Known bad nodes"). Resolved by excluding them
  (`--exclude=nodo3,nodo5`); predict/slim/features then completed. Earlier the run
  was also blocked on the ColabFold MMseqs2 MSA server rejecting requests — that
  cleared and MSAs are now on disk under `data/processed/fireprot_le200/msas/`.

## Log — newest first
### 2026-07-18 — new error-vs-ΔΔG diagnostic; nodo1 found bad; ≤500 predict resubmitted
- Dropped the linear-calibration idea (didn't add skill). Added **in-range vs
  out-of-range** split to `transfer.py` (by training ΔΔG range, default 1st/99th pct;
  used [-1,4]): per-bin r/ρ/RMSE/MAE + a scatter that shades the in-range band and
  colors out-of-range points. Key ≤200 finding: in-range r=0.66 RMSE=0.87; out-of-range
  RMSE≈3.7 and the within-tail correlation collapses (below -1: r=-0.61; above 4: r=0.21)
  — the pooled out-of-range r is a two-cluster artifact (kept out of the legend).
- Added `ddg/evaluation/error_curves.py`: prediction error/variance vs measured ΔΔG
  (binned bias±SD + RMSE/MAE). Shows the regression-to-mean signature for both the
  transfer (05) and within-Tsuboyama (01) — error is bias-dominated, rising toward both
  tails. `transfer.py` now emits `error_vs_ddg.png` too. TODO: roll the error curve out
  to the other experiments (need their prediction parquets; 02/03/06 are cluster-only).
- **`fireprot_201to500` predict failed**: 7/8 shards landed on **nodo1**, which crashes
  boltz at CUDA init (`RuntimeError: CUDA unknown error … Setting the available devices
  to be zero`). New bad node — added to CLAUDE.md + exclude list. Cancelled the failed
  chain (212246 FAILED, 212247/212248 CANCELLED); **resubmitted predict 212256 → slim
  212257 → features 212258** with `--exclude=nodo1,nodo3,nodo5,…`. prepare 212245 was
  fine (1,715 queries + MSAs on disk), so predict-only resubmit.
### 2026-07-18 — extending to ≤500 aa: running the 201–500 band separately
- Goal: extend the FireProt transfer test from ≤200 aa to **≤500 aa** by running only
  the **excluded band** (201–500 aa) and merging with the existing `fireprot_le200`
  features (proteins are disjoint by length, so the merge is a clean concat).
- Band source: `data/raw/fireprot_201to500.csv` = `fireprot_filtered.csv` sliced to
  201≤seqlen≤500 → **1,662 muts / 53 proteins** (all valid: 0 wt/pos mismatch, 0 NaN
  id, 0 overlap with the 85 le200 proteins). Config `experiment_configs/fireprot_201to500.yaml`
  (clone of le200). Committed `15e00c1`, pushed, pulled on cluster.
- Combined ≤500 target = 85 + 53 = **138 proteins / 3,205 muts** (note: this keeps the
  3 UniProt-less proteins that `fireprot_filtered_500.csv` had dropped).
- Submitted chain: **prepare 212245 → predict 212246** (8 shards, `%3`,
  `--exclude=nodo3,nodo5,nodo11,nodo12,nodo14,nodo15,sauron`) **→ slim 212247 → features 212248**.
  Watch: prepare fetches **53** new base MSAs from the ColabFold server (bigger load
  than the 3 before); proteins are larger (mean 371 aa, max 477) so predict is heavier.
- Next when features land: concat the two `features_summary.parquet` → `fireprot_le500`
  merged table, re-run `ddg.evaluation.transfer` on the ≤500 corpus, compare to ≤200.
### 2026-07-18 — added report.pdf
- Wrote `build_report.py` (self-contained; reads the committed summaries/per_protein
  + processed parquets, embeds figures as base64, renders via `wkhtmltopdf` — no LaTeX)
  and committed `report.pdf` (4 pp: abstract, methods, results w/ both figures + 3
  tables, the magnitude ceiling, corpus-completeness provenance).
### 2026-07-18 — full corpus extracted; transfer eval done → result complete
- Chain 212209–212212 all COMPLETED. Pulled `features_summary.parquet`:
  **1,543 muts / 85 proteins**, all 3 recovered proteins present, no NaN features.
- Ran `python -m ddg.evaluation.transfer --train …/tsuboyama_bench_fast/rawz_features.parquet
  --test …/fireprot_le200/features_summary.parquet --model mlp` (also `--model hgb`).
  **MLP:** pooled r=0.621, ρ=0.684, RMSE=1.41, MAE=0.86; per-protein mean 0.488 /
  **median 0.668** (76/85 scored). **HGB:** r=0.607, ρ=0.675. `sign_flipped=false`.
  Predicted-vs-measured slope 0.262 (magnitude compression, as in 02/06).
- Outputs: `data/processed/fireprot_le200/transfer_from_tsuboyama{,_hgb}/`. Copied
  tables + scatter into this folder; built `figures/02_per_protein_r_hist.png`; wrote
  `README.md` + `figures/README.md`; updated `results/README.md` + `history.md`.
  State → ✅ Done.
### 2026-07-18 — root-caused the 29-mutation gap; adapter fix + full re-run
- First re-run (jobs 212187/212188/212189) completed and lifted coverage to
  **1,514 muts / 82 proteins** (from 773/54) — but still 29 short of 1,543.
- Root cause of the 29: the FireProt adapter keyed `wt_id` on `uniprot_id` alone.
  **3 proteins have no UniProt id** — `3PG0` (ThreeFoil, 10 muts), `2IMM` (5),
  `1YYX` (14) — only a `pdb_id` + sequence, so `wt_id` was NaN and they never got
  MSAs/queries. Not a compute failure; positions are all in range.
- Fix (commit `5e55812`): `dataset_fireprot.get_wt_id` falls back to `pdb_id` when
  `uniprot_id` is missing → 0 NaN wt_id, **85 proteins / 1,543 muts** (verified
  locally through the adapter). Pushed; pulled on cluster.
- Resubmitted the **full** chain (prepare needed to add the 3 proteins' MSAs/queries;
  `overwrite: false` so the 82 existing MSAs are reused, only 3 new base MSAs fetched):
  **prepare 212209 → predict 212210** (8 shards, `%3`, `--exclude=nodo3,nodo5,nodo11,nodo12,nodo14,nodo15,sauron`)
  **→ slim 212211 → features 212212**. Watch: prepare hits the ColabFold MSA server
  for the 3 new WT sequences.
- Transfer eval (`ddg/evaluation/transfer.py`, default model now `mlp`) will run on the
  full 1,543-row features once the chain lands.
### 2026-07-18 — discovered partial features; re-running full extraction
- Audited the actual `features_summary.parquet`: **773 muts / 54 proteins**, not the
  1,543 / 82 that the prior status entry claimed. Feature columns are the raw-Δz set
  (256 `zdiag_*`/`zpool_*`), identical to Tsuboyama `rawz_features.parquet`; ddg sign
  convention matches (both ~75% positive = destabilizing, so **no sign flip**); zero
  `wt_id` overlap with Tsuboyama (clean independent test).
- Manifest has no `predict` step → slim ran on a half-finished predict array; with
  `delete_raw: true` the raw NPZs are gone, so the missing structures can't resume.
- User wants the full corpus → resubmitted a fresh **predict → slim → features** chain
  on the cluster (skipping prepare; MSAs+1,597 queries already on disk, so the MSA
  server is never touched). Jobs: **predict array 212187** (8 shards, `%3`), **slim
  212188**, **features 212189**; predict `--exclude=nodo3,nodo5,nodo11,nodo12,nodo14,nodo15,sauron`.
- Wrote `ddg/evaluation/transfer.py` (cross-dataset transfer eval: train on all of one
  raw-Δz table, predict an independent one; pooled + per-protein metrics, auto
  sign-flip guard, scatter). Not yet committed. Transfer eval will run on the FULL
  FireProt features once the chain finishes.
### 2026-07-18 — state audit
- Confirmed on the cluster: FireProt features complete (`features_summary.parquet`,
  1,543 muts / 82 proteins), **no benchmark output**. The transfer eval is the
  remaining step. Created this folder + status to stop the work getting lost.
  (The eval jobs running today — 212168/212172/212173 — are all `tsuboyama_bench_fast`,
  not FireProt.)
### 2026-07-17 — pipeline completed
- prepare → predict → slim → features finished for `fireprot_le200` after excluding
  bad nodes nodo3/nodo5 from the predict array; `features_summary.parquet` written.
