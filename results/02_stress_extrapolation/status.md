# Status — 02_stress_extrapolation

**State:** ✅ Done
**Last updated:** 2026-07-17

## Current state
Complete. Extrapolation-to-the-tail stress test on the wide corpus
(`data/processed/tsuboyama_bench_wide/`, 37,080 mutations, raw Δz, HGB): train on
mild mutations (|ΔΔG|<1), test on the strongly destabilizing tail (ΔΔG>2). Result:
the model collapses out of range — tail r ≈ 0.09, fit slope ≈ 0.02, predictions cap
near the training range. Numbers in `extrapolation_summary.{csv,json}`, figure
`extrapolation_pred_vs_actual.png`.

## Next steps
- [ ] None — result settled. Documents a known limitation (no extrapolation beyond
      trained ΔΔG range); revisit only if the model/objective changes to target it.

## Log — newest first
### 2026-07-17 — completed
- Ran the tail-extrapolation stress test; packaged summary + figure. See
  `history.md` §4.
