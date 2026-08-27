# figures — 10_gla_scan

Regenerate everything from committed scripts (no cluster access needed once
`scan_predictions.csv` exists):

```bash
python results/10_gla_scan/compare_foldx.py --scan <scan_predictions.csv>   # 01 + percentile_shift_mean.csv
python results/10_gla_scan/map_discrepancy.py --pdb <1r46.pdb>              # 02, 03
python results/10_gla_scan/compare_lukas.py                                 # 04
```

| Figure | What it shows | Read it as |
|---|---|---|
| **`01_heatmap_mean.png`** | Predicted ΔΔG for every substitution at each scanned position (mean of regimes A/B/D). Rows = mutant residue grouped by chemistry; columns = the scanned sites. | The deliverable. Grey = no value (wild-type cell, or not computed). Columns are **not contiguous in sequence** — 38 selected sites. |
| **`01_boltz_vs_foldx_mean.png`** | Three panels. Top left: per-mutation Boltz vs FoldX, ρ=+0.595, symlog x (FoldX runs to +69). **Top right: the same points in percentile space** — each method ranked within its own spread, with the `pct(Boltz)=pct(FoldX)` diagonal and the share of each group below it. Bottom: per-position means, flagged sites shaded, glycines marked ▲. | The comparison. The percentile panel needs **no fitted line** — an OLS fit has slope 0.055, so "below the line" would collapse into "low Boltz" (glycines 58% vs 56% for the rest, against 78% vs 38% here). Below the diagonal = FoldX ranks it higher within its own spread. |
| **`02_discrepancy_map.png`** | Disagreement in **relative ordering**: `pct(Boltz) − pct(FoldX)` per position, with burial underneath. | **Not an accuracy measure** — there is no ground truth. Percentiles are within this glycine-heavy set, and rank-normalising hides the scale mismatch by construction. |
| **`03_discrepancy_map_raw.png`** | The same comparison in **real kcal/mol**. Top: raw difference. Bottom: both methods' per-position means side by side. | The scale mismatch. The difference correlates −0.975 with FoldX alone; FoldX varies 10.3× more than Boltz. |

| **`04_lukas_activity.png`** | Predicted ΔΔG (left: Boltz, right: FoldX) vs **measured** residual α-Gal A activity in HEK293H (Lukas 2013). **n = 41** — the 45 shared mutations minus the 4 at active-site positions (`C142R`, `A143P`, `A143T`, `D170N`), which are excluded because a catalytic variant can be dead while folded. Red bars = mean activity per tercile of the prediction, labelled with the count at exactly 0 %. FoldX axis is symlog. | The only external measurement that touches this scan. Both get the right sign (ρ = −0.328 / −0.343) and are indistinguishable (paired CI [−0.25, +0.29]). **Activity ≠ stability** — a weak ordinal check, not accuracy. |

Annotation convention shared by 01–03: **gold shading + bold `*` label** = one of the
10 positions flagged as overestimated; **black outline / ▲ / `G` strip** = wild-type
glycine. They are separate channels because 3 positions (80, 325, 360) are both.

Not committed but reproducible: `boltz_minus_foldx.pdb` (B-factor = signed ordering
disagreement) for viewing the map in 3D — see `map_discrepancy.py`.
