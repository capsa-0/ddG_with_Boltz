# Figures — 06_mlp_generalization

| File | Content |
|---|---|
| `01_mlp_vs_hgb_holdouts.png` | **Headline.** Grouped bar of pooled Pearson r per holdout, MLP (5-seed ensemble) vs the HGB baseline from experiment 01. |
| `02_holdout_pearson_bar.png` | MLP pooled r per holdout (eval-generated, MLP alone). |
| `03_per_protein_distribution.png` | Per-protein r distribution (protein holdout), MLP. |
| `04_scatter_random.png` | Predicted vs experimental ΔΔG, random holdout (MLP). |
| `05_scatter_protein.png` | Predicted vs experimental ΔΔG, protein holdout (MLP). |
| `06_substitution_heatmap.png` | 20×20 source×target leave-one-substitution-out r (MLP). |
| `07_chemistry_bar.png` | Per-chemistry-class r, introduce vs remove (MLP). |
| `08_error_vs_ddg.png` | Prediction error vs measured ΔΔG (random holdout, MLP): bias ± SD (top), RMSE/MAE with bin counts (bottom). Regression-to-mean — bias crosses zero near the training mode, error rises toward both tails. |
| `09_density_vs_error.png` | Test error vs **training density** in ΔΔG space. Left: density and error are mirror images along ΔΔG. Right: error vs training density (log-x), Spearman ρ = **−0.97** (bins) / −0.40 (points) — error is governed by how densely Tsuboyama sampled that ΔΔG. |

Figures 02–07 are produced by `ddg.evaluation` (`plots.py`) for the MLP run;
figure 01 is the cross-model comparison built for this write-up. Figures 08–09 are
from `ddg.evaluation.error_curves` (random-holdout OOF predictions; training density =
Tsuboyama ΔΔG).
</content>
