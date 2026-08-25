# figures — 10_gla_scan

Regenerate everything from committed scripts (no cluster access needed once
`scan_predictions.csv` exists):

```bash
python results/10_gla_scan/compare_foldx.py --scan <scan_predictions.csv>   # 01
python results/10_gla_scan/map_discrepancy.py --pdb <1r46.pdb>              # 02, 03
```

| Figure | What it shows | Read it as |
|---|---|---|
| **`01_heatmap_mean.png`** | Predicted ΔΔG for every substitution at each scanned position (mean of regimes A/B/D). Rows = mutant residue grouped by chemistry; columns = the scanned sites. | The deliverable. Grey = no value (wild-type cell, or not computed). Columns are **not contiguous in sequence** — 38 selected sites. |
| **`01_boltz_vs_foldx_mean.png`** | Left: per-mutation Boltz vs FoldX, ρ=+0.504, symlog x (FoldX runs to +69). Right: per-position means, flagged sites shaded, glycines marked ▲. | The comparison. Note ρ *falls* to +0.379 once FoldX's clash tail is excluded. |
| **`02_discrepancy_map.png`** | Disagreement in **relative ordering**: `pct(Boltz) − pct(FoldX)` per position, with burial underneath. | **Not an accuracy measure** — there is no ground truth. Percentiles are within this glycine-heavy set, and rank-normalising hides the scale mismatch by construction. |
| **`03_discrepancy_map_raw.png`** | The same comparison in **real kcal/mol**. Top: raw difference. Bottom: both methods' per-position means side by side. | The scale mismatch. The difference correlates −0.975 with FoldX alone; FoldX varies 10.3× more than Boltz. |

Annotation convention shared by 01–03: **gold shading + bold `*` label** = one of the
10 positions flagged as overestimated; **black outline / ▲ / `G` strip** = wild-type
glycine. They are separate channels because 3 positions (80, 325, 360) are both.

Not committed but reproducible: `boltz_minus_foldx.pdb` (B-factor = signed ordering
disagreement) for viewing the map in 3D — see `map_discrepancy.py`.
