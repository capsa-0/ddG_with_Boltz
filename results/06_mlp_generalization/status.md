# Status — 06_mlp_generalization

**State:** 🚧 In progress
**Last updated:** 2026-07-18

## Current state
Replication of `01_generalization` with an **MLP** regressor in place of HGB, same
corpus/features/holdouts (`tsuboyama_bench_fast`, 12,359 muts, 256 raw-Δz features,
`rawz_features.parquet`). The `mlp` estimator was redesigned (commit `215a6ce`) from
a lone single-seed `MLPRegressor((256,64))` — which was unstable across group folds
(non-monotonic homology) — to a **5-seed `VotingRegressor`** of a deeper,
L2-regularized MLP `(256,128,64), alpha=3e-3, patient early stopping`, fit in
parallel. The discarded single-seed variant is not reported.

Homology sweep done; main 7-holdout job still running (writing at the very end).

## Next steps
- [ ] Main job 212168 finishes → pull `benchmark_rawz_mlp/benchmark_summary.csv`.
- [ ] Build MLP-vs-HGB comparison figure + README + figures/README index.
- [ ] Update top-level `results/README.md` index + `history.md`.

## Blockers
- None.

## Log — newest first
### 2026-07-18 — MLP replication run
- Redesigned `ddg/evaluation/models.py` `mlp` → 5-seed VotingRegressor (commits
  `0845ba2`, `215a6ce`). First single-seed attempt gave unstable homology
  (0.47/0.39/0.72) → cancelled and redone.
- Cluster jobs (cpu, `--exclude=nodo3,nodo5`, 6 cpus, BLAS-thread-capped) on
  `rawz_features.parquet`, out dirs `benchmark_rawz_mlp[_clNN]`:
  - Homology **done**: cl30=0.781, cl50=0.785, cl90=0.790 (jobs 212172/73/74).
    (HGB baseline: 0.765/0.766/0.772.)
  - Main 7-holdout **running** (job 212168, nodo1). Partial: random 0.803,
    protein 0.792, denovo 0.703; substitution/source/target/chemistry pending.
- Bad-node incidents: 212171 SIGSEGV on nodo5; 212169/70 landed on nodo3 (both
  cancelled+resubmitted with `--exclude`). Reminder: exclude nodo3,nodo5 on eval too.
