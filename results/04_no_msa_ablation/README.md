# 04 — No-MSA ablation (does the evolutionary signal matter?)

**What:** How much of the ΔΔG signal comes from Boltz's **multiple-sequence
alignment (MSA)** input vs. its structural prior? We re-ran the fast corpus with
Boltz in **single-sequence mode** (`msa: empty`) and compared it, holdout-by-holdout,
against the MSA run.

**Why:** Boltz normally sees an MSA per protein — an evolutionary signal that is
itself informative about stability. Removing it isolates what the model's
structural prior alone contributes, and tells us whether the pipeline depends on a
(rate-limited, sometimes-unavailable) MSA server.

**How:** Identical corpus, features, and model to study 01 — the **only** difference
is the MSA. `no_msa: true` makes `prepare` skip the MMseqs2 search and emit
`msa: empty` in every Boltz query; everything downstream (raw Δz, HGB, holdout
suite) is unchanged. Both runs are z-only (256 raw-Δz features, `--drop-s`).

## Result — MSA vs. no-MSA (pooled Pearson r)

| Holdout | MSA | no-MSA | Δr | MSA RMSE | no-MSA RMSE |
|---|---|---|---|---|---|
| random | 0.783 | 0.699 | **−0.084** | 0.632 | 0.726 |
| protein | 0.774 | 0.687 | −0.087 | 0.643 | 0.738 |
| de-novo (natural↔designed) | 0.705 | 0.606 | **−0.099** | 0.724 | 0.812 |
| substitution | 0.772 | 0.691 | −0.081 | 0.642 | 0.730 |
| source_residue | 0.754 | 0.670 | −0.085 | 0.667 | 0.755 |
| target_residue | 0.743 | 0.664 | −0.079 | 0.680 | 0.760 |
| chemistry | 0.734 | 0.647 | −0.087 | 0.691 | 0.775 |
| homology (30 % id) | — | 0.682 | — | — | 0.743 |

*(The MSA raw-Δz benchmark did not include the homology holdout, so Δ is blank
there; the no-MSA value is reported for completeness.)*

**The MSA is worth a consistent ~0.08–0.10 r.** Removing it costs **0.08–0.10
pooled Pearson** and ~0.09 kcal/mol RMSE on *every* holdout — a strikingly uniform
penalty (mean Δr ≈ −0.086). Two takeaways:
1. **The structural prior alone is already strong** — single-sequence Boltz still
   reaches r = 0.70 (random) / 0.69 (unseen proteins). Most of the ΔΔG signal is
   structural, not evolutionary.
2. **The MSA adds a real, non-trivial boost** on top of that, largest for **de-novo
   transfer** (−0.099) — predicting designed proteins leans most on the evolutionary
   input. So keep the MSA when it's available, but the predictor degrades gracefully
   (not catastrophically) without it.

See `comparison.png`.

## Data & provenance

| Item | MSA run | no-MSA run |
|---|---|---|
| Config | `experiment_configs/tsuboyama_bench_fast.yaml` | `experiment_configs/tsuboyama_bench_fast_nomsa.yaml` |
| Processed dir | `data/processed/tsuboyama_bench_fast/` | `data/processed/tsuboyama_bench_fast_nomsa/` |
| Benchmark output | `.../benchmark_rawz/benchmark_summary.csv` | `.../benchmark_no_s/benchmark_summary.csv` |
| Corpus | Tsuboyama fast, 12,359 mutations, 412 proteins (same for both) | |
| Model / features | HGB on 256 raw-Δz (z-only, `--drop-s`) | same |
| Homology map | reused `data/processed/tsuboyama_bench_fast/cluster_map_30.csv` (same proteins) | |

Reproduce the comparison:
```
python -m ddg.evaluation.compare_runs \
  --a <MSA>/benchmark_rawz/benchmark_summary.csv --label-a MSA \
  --b <no-MSA>/benchmark_no_s/benchmark_summary.csv --label-b no-MSA \
  --out results/04_no_msa_ablation
```

## Files
- `comparison.png` — grouped bar chart, MSA vs. no-MSA per holdout, with Δr labels.
- `comparison.csv` — merged table (r / RMSE / MAE for both + Δr).
- `benchmark_summary_nomsa.csv` — the raw no-MSA benchmark summary (all metrics).
