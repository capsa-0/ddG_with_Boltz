# figures — 12_error_anatomy

| File | What it shows |
|---|---|
| `01_s669_mut_class_error.png` | **S669** (regime D, 541 variants). Left: protein-centred error by Gly/Pro class. Middle: by burial tertile (Boltz distogram contact number). Right: predicted vs true ΔΔG coloured by effect direction, with the amplitude-compression fit. |
| `02_tsuboyama_mut_class_error.png` | Same three panels on **12,359 out-of-fold Tsuboyama** predictions (5-fold GroupKFold on `wt_id`). The large n is what makes the class comparisons readable; S669's Gly/Pro cells are too small to interpret. |

Both figures plot the **protein-centred** error (per-protein mean removed) so that
class effects are not confounded with the cross-protein calibration gap analysed in
`results/11_calibration_gap`.
