# Status — 03_stress_learning_curve

**State:** ✅ Done
**Last updated:** 2026-07-17

## Current state
Complete. Learning curve (pooled r vs. number of training proteins, proteins held
out) on the wide corpus (`data/processed/tsuboyama_bench_wide/`, 37,080 mutations,
raw Δz, HGB). Result: near-saturated — 33 proteins already give r ≈ 0.74; 10× more
(330) adds only ~0.05 (→ 0.79). Numbers in `learning_curve.csv`, figure
`learning_curve.png`.

## Next steps
- [ ] None — result settled.

## Log — newest first
### 2026-07-17 — completed
- Ran the learning-curve sweep; packaged CSV + figure. See `history.md` §4.
