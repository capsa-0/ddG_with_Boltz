# Status — 04_no_msa_ablation

**State:** ✅ Done
**Last updated:** 2026-07-17

## Current state
Complete. MSA vs. single-sequence Boltz ablation: fast corpus re-run with Boltz in
single-sequence mode (`no_msa: true`, `experiment_configs/tsuboyama_bench_fast_nomsa.yaml`),
identical corpus/features/model, differing only in the MSA. Result: the MSA is worth
a uniform ~0.08–0.10 pooled r across every holdout (mean Δr ≈ −0.086, largest for
de-novo, −0.099); single-sequence Boltz still reaches r 0.70 — most of the ΔΔG
signal is structural. Numbers in `comparison.csv` / `benchmark_summary_nomsa.csv`,
figure `comparison.png`.

## Next steps
- [ ] None — result settled.

## Log — newest first
### 2026-07-17 — completed
- Ran the no-MSA ablation and the paired comparison against 01; packaged tables +
  figure. See `history.md` §5.
