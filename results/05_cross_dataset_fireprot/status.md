# Status — 05_cross_dataset_fireprot

**State:** 🚧 In progress — features were only PARTIAL (773/1543); full re-extraction
running on the cluster, transfer eval will run on the full corpus once it lands
**Last updated:** 2026-07-18

## Current state
Cross-dataset transfer test: does the **Tsuboyama-trained** raw-Δz ΔΔG predictor
generalize to the independent **FireProt** dataset?

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
- **Cluster processed dir:** `data/processed/fireprot_le200/` — has `msas/`,
  `queries/`, `boltz_raw_output/`, `slim/`, and **`features_summary.parquet`**
  (built 2026-07-17 15:39). prepare → predict → slim → features are DONE.
- **Missing:** no `benchmark/` output under `fireprot_le200`. The transfer eval
  (train on Tsuboyama features → predict FireProt → correlate) has not been wired up
  or run. Note: `python -m ddg.evaluation` does *within-dataset* holdouts, so running
  it on this config alone gives a FireProt-internal benchmark, **not** the transfer
  result we want.

## Next steps
- [ ] Wire/run the transfer evaluation: fit the raw-Δz model on Tsuboyama
      (`tsuboyama_bench_fast` or `_wide` `rawz_features.parquet`), predict on
      `data/processed/fireprot_le200/features_summary.parquet`, report pooled r /
      RMSE / MAE (overall + per-protein).
- [ ] **Watch the ΔΔG sign convention** — FireProt (`ddG`) vs Tsuboyama (`ddg`) may
      differ in which sign is destabilizing. If correlation comes out negative, flip.
      (See `CLAUDE.md` "ΔΔG column name differs by adapter".)
- [ ] Once there are numbers: write `README.md` (What/Why/How + provenance table +
      headline), add `figures/`, and update `results/README.md` + `history.md`
      (move 05 from "Planned" to a result row).

## Blockers
- None active. **Historical:** the predict step repeatedly failed at startup when
  shards landed on bad GPU nodes `nodo3` / `nodo5` (SIGSEGV / ld.so crash — see
  CLAUDE.md "Known bad nodes"). Resolved by excluding them
  (`--exclude=nodo3,nodo5`); predict/slim/features then completed. Earlier the run
  was also blocked on the ColabFold MMseqs2 MSA server rejecting requests — that
  cleared and MSAs are now on disk under `data/processed/fireprot_le200/msas/`.

## Log — newest first
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
