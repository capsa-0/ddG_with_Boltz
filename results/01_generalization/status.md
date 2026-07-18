# Status — 01_generalization

**State:** ✅ Done
**Last updated:** 2026-07-17

## Current state
Complete. Generalization-holdout study of the raw-Δz ΔΔG predictor on the Tsuboyama
`tsuboyama_bench_fast` corpus (12,359 mutations, HGB on 256 raw-Δz features).
Headline: random-CV pooled r = 0.783, protein-holdout 0.702, homology (30%) 0.765,
per-protein mean r 0.806. Report, figures, and provenance table are in `README.md` /
`details.md` / `report.pdf`. Benchmark output on cluster:
`data/processed/tsuboyama_bench_fast/benchmark_rawz/`.

## Next steps
- [ ] None — result settled. (Re-open only if the feature set or corpus changes.)

## Log — newest first
### 2026-07-17 — folder finalized
- Restructured into the one-folder-per-result layout; added the explicit
  data/config/provenance table to the README.
### 2026-07-16 — study completed
- Added homology (identity-threshold) + per-residue holdouts, RMSE/MAE columns;
  established raw Δz as the representation and that the model generalizes across
  proteins. See `history.md` §1–§3 for the narrative.
