# figures — 05_cross_dataset_fireprot

Transfer of the Tsuboyama-trained raw-Δz predictor (MLP) to FireProt, n=1,543 / 85 proteins.

- **`01_transfer_scatter.png`** — predicted (trained on Tsuboyama) vs measured ΔΔG
  (FireProt). Pooled r=0.62, ρ=0.68, RMSE=1.41. Shows the ranking signal **and** the
  magnitude compression: the cloud is far flatter than the y=x diagonal (fit slope
  ≈ 0.26) — the model under-predicts the destabilizing/stabilizing tails.
- **`02_per_protein_r_hist.png`** — distribution of per-protein Pearson r (76 of 85
  proteins scored; the rest have <2 mutations or constant ΔΔG). Concentrated at
  0.6–1.0 (median 0.67); mean 0.49 is dragged down by a handful of poorly-transferring
  proteins.
</content>
