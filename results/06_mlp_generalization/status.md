# Status — 06_mlp_generalization

**State:** ✅ Done
**Last updated:** 2026-07-18

## Current state
Complete. Replication of `01_generalization` with an **MLP** (5-seed
`VotingRegressor` of `(256,128,64), alpha=3e-3` nets, commit `215a6ce`) in place of
HGB — same corpus/features/holdouts (`tsuboyama_bench_fast`, 12,359 muts, 256 raw-Δz
features). **Headline: MLP matches/slightly beats HGB on every holdout** (random
0.803, protein 0.792, homology 0.781/0.785/0.790, per-protein mean 0.827) → the
generalization is a property of the raw-Δz representation, not the model. README,
details, figures, combined `benchmark_summary.csv` all committed. Benchmark output on
cluster: `data/processed/tsuboyama_bench_fast/benchmark_rawz_mlp[_cl{30,50,90}]/`.

## Next steps
- [ ] None — result settled. (Follow-up, not blocking: parallelize the eval fold
      loop, `ddg/evaluation/benchmark.py`, to cut the 332-fold substitution cost.)

## Blockers
- None.

## Log — newest first
### 2026-07-18 — completed + written up
- Main job 212168 COMPLETED (~75 min). Full MLP numbers pulled; MLP ≥ HGB on all
  holdouts. Built comparison figure `figures/01_mlp_vs_hgb_holdouts.png`, README,
  details.md, figures index; updated top-level `results/README.md` + `history.md`.
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
