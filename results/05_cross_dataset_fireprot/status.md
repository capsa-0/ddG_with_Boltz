# Status — 05_cross_dataset_fireprot

**State:** 🚧 In progress — features done, transfer evaluation not yet run
**Last updated:** 2026-07-18

## Current state
Cross-dataset transfer test: does the **Tsuboyama-trained** raw-Δz ΔΔG predictor
generalize to the independent **FireProt** dataset? The Boltz feature pipeline for
FireProt is **complete on the cluster**; the actual transfer evaluation has **not**
been run and there are no numbers yet.

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
### 2026-07-18 — state audit
- Confirmed on the cluster: FireProt features complete (`features_summary.parquet`,
  1,543 muts / 82 proteins), **no benchmark output**. The transfer eval is the
  remaining step. Created this folder + status to stop the work getting lost.
  (The eval jobs running today — 212168/212172/212173 — are all `tsuboyama_bench_fast`,
  not FireProt.)
### 2026-07-17 — pipeline completed
- prepare → predict → slim → features finished for `fireprot_le200` after excluding
  bad nodes nodo3/nodo5 from the predict array; `features_summary.parquet` written.
