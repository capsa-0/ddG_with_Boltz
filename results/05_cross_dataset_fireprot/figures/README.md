# figures — 05_cross_dataset_fireprot

Tsuboyama-trained raw-Δz predictor (MLP) transferred to FireProt ≤500 (3,205 muts / 138 proteins).

- **`01_transfer_scatter.png`** — predicted vs measured ΔΔG; in-range band shaded, out-of-range
  points colored. Pooled r=0.65; the cloud is flatter than y=x (magnitude compression).
- **`02_per_protein_r_hist.png`** — per-protein Pearson r (114/138 scored); median 0.65.
- **`03_error_vs_ddg.png`** — prediction error vs measured ΔΔG: regression-to-mean bias, error
  minimized in the dense centre, rising toward both tails.
- **`04_density_vs_error.png`** — test error vs Tsuboyama training density; Spearman ρ=−0.96.
- **`05_residual_vs_ddg.png`** — per-mutation residual (predicted − experimental) vs experimental
  ΔΔG; trend slope ≈ −0.73 (systematic under-prediction of large effects). In-range near 0,
  out-of-range fan out.
