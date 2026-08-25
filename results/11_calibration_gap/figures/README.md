# figures — 11_calibration_gap

| File | What it shows |
|---|---|
| `01_per_protein_error.png` | **S669**, all three training regimes. Left: distribution of per-protein mean *signed* error — the offset term (sd 1.41–1.55 kcal/mol, range −6.1 to +3.6). Middle: per-protein MAE (medians B 0.85 / D 1.05 / A 1.12). Right: per-protein mean error vs the protein's own true mean ΔΔG — **r = 0.91, slope 0.77**, i.e. the model predicts nearly the same mean for every protein, so the offset it needs is essentially the protein's true mean ΔΔG. |

Pending: a panel contrasting the S669 offset ceiling (+0.204) with the held-out
Tsuboyama ceiling (+0.029) — the comparison that carries the folder's conclusion.
