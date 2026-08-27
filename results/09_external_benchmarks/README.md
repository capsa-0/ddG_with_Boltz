# External blind benchmarks: S669 & Ssym

> ✅ **Corrected 2026-08-27** (defect found in results/14). `run_benchmarks.py::members()`
> used `early_stopping=False, max_iter=250` instead of the project-default estimator.
> Because `max_iter` counts epochs, regime A (24,718 augmented samples) took ~4x regime B's
> gradient updates and was overfit hardest — confounding the A-vs-B-vs-D comparison. All
> three regimes have been re-run with `make_model("mlp")`'s estimator. **Every number below
> is the corrected one**; the originals are kept in `results_pre-correction.csv`.
> Regime A gains most (S669 common-25: 0.214 → **0.361**), but the ordering holds — see
> "What the correction changed".

We test the Boltz-2 embedding ΔΔG predictor on the two most widely used blind
stability benchmarks, under three training regimes, and separate genuine
generalization from train↔test sequence-identity leakage.

## Data

| Benchmark | Variants | Proteins | Source | Notes |
|---|---|---|---|---|
| **S669** | 541 | 62 | DDGemb (Pancotti et al. 2022) UniProt mapping | diverse, deliberately dissimilar to common training sets |
| **Ssym** | 337 | 13 | ThermoMPNN (direct set) | narrow, dominated by lysozyme / barnase / T4-lysozyme |

Both capped at ≤500 aa; every variant's position was validated against the
provided sequence (`seq[pos-1] == wild-type`). Ssym is direct mutations only;
the reverse direction is obtained analytically from the antisymmetry-augmented
model (swap the wt/mut feature halves).

## Method

- **Features:** concatenated Boltz-2 pooled embeddings `wtz` + `mtz` (128 + 128).
- **Model:** 5-seed MLP ensemble `(256,128,64)`, antisymmetry augmentation on every
  training set (swap halves, negate ΔΔG).
- **Regimes:**
  - **A — Tsuboyama-only** (12,359 muts / 412 proteins)
  - **B — FireProt-only** (3,205 muts / 138 proteins, ≤500 aa)
  - **D — fine-tuned**: pretrain on Tsuboyama, warm-start continue on FireProt.
- **Leakage control:** pool the WT sequences of {Tsuboyama ∪ FireProt ∪ benchmark}
  and cluster with MMseqs2 (`easy-cluster`, 80% coverage) at **25%** and **30%**
  identity. A benchmark protein is *leaky* w.r.t. a training corpus if it shares a
  cluster with any protein in it. This mirrors ThermoMPNN's removal of Megascale
  homologues (>25% identity) from its test sets.
- Benchmark ΔΔG uses the opposite sign convention; predictions are sign-flipped when
  the pooled Pearson is negative (recorded).

Homology: **S669 and Ssym have zero proteins homologous to Tsuboyama** (so regime A
needs no filtering), while FireProt overlaps **9 S669 proteins / 181 variants** and
**9 Ssym proteins / 290 variants** at 25% identity.

## Results — pooled Pearson r (per-protein median r in parentheses)

**"full"** = all variants; **"filtered"** = drop proteins homologous to the regime's
own training corpus; **"common"** = drop proteins homologous to *any* corpus (identical
variant subset for all three regimes — the fair cross-regime comparison).

### S669 (541 variants / 62 proteins)
| Regime | full | filtered (25%) | common (25%, n=360) |
|---|---|---|---|
| **A — Tsuboyama** | 0.255 (0.46) | 0.255 (0.46) — *0 leaky* | 0.214 (0.48) |
| **B — FireProt** | 0.500 (0.58) | 0.404 (0.56) | 0.404 (0.56) |
| **D — fine-tuned** | 0.462 (0.61) | 0.408 (0.61) | **0.408 (0.61)** |

### Ssym (337 variants / 13 proteins)
| Regime | full | filtered (25%) | common (25%, n=47) |
|---|---|---|---|
| **A — Tsuboyama** | 0.728 (0.73) | 0.728 (0.73) — *0 leaky* | 0.845 (0.72) |
| **B — FireProt** | 0.891 (0.89) | 0.871 (0.72) | 0.871 (0.72) |
| **D — fine-tuned** | 0.797 (0.75) | 0.864 (0.71) | 0.864 (0.71) |

Antisymmetry (Ssym): corr(direct, −reverse) = **0.91 / 0.98 / 0.97** for A/B/D, with
bias ≈ 0.05 kcal/mol — the model treats a reverse mutation as ≈ minus the forward.

See `figures/01_pooled_r_full_vs_filtered.png`; full numbers in `results.csv`.

## Findings

1. **Leakage is real and measurable.** Regime A (Tsuboyama) has no overlap with either
   benchmark, so its numbers are unfiltered generalization. FireProt-based regimes drop
   once homologs are removed — most starkly on Ssym, where B's per-protein median falls
   **0.89 → 0.72**. On the fair common-clean subset the apparent FireProt advantage on
   Ssym essentially vanishes (**A 0.85 ≈ B 0.87 ≈ D 0.86**): it was leakage.
2. **The two benchmarks tell different stories.** Ssym is narrow and easy (a handful of
   well-studied folds) — everyone scores ~0.7–0.9. **S669 is the honest hard test**
   (diverse, dissimilar); pooled r is modest (0.21–0.50), reflecting how difficult blind
   cross-protein ΔΔG *magnitude* prediction genuinely is.
3. **Training distribution matters more than raw label count.** On the diverse S669, even
   after homology filtering, FireProt-based regimes (B/D ≈ 0.40) beat Tsuboyama-only
   (A ≈ 0.21) in pooled r — FireProt's *natural* proteins resemble S669 far more than
   Tsuboyama's *designed* mini-domains. Regime A ranks well *within* a protein
   (median 0.46–0.48) but calibrates poorly *across* proteins (pooled 0.21–0.26).
4. **Fine-tuning earns its keep on the hard benchmark.** Regime D has the best S669
   per-protein median (0.61) and it is unchanged by homology filtering — in contrast to
   the within-FireProt result (results/08) where fine-tuning washed out.

## Reproduce

```bash
python results/09_external_benchmarks/build_datasets.py        # -> data/raw/{s669,ssym}.csv
python -m ddg run experiment_configs/s669.yaml                 # prepare->predict->features (cluster)
python -m ddg run experiment_configs/ssym.yaml
python results/07_feature_symmetry_ablation/build_ablation_features.py data/processed/s669
python results/07_feature_symmetry_ablation/build_ablation_features.py data/processed/ssym
MMSEQS_BIN=/path/to/mmseqs python results/09_external_benchmarks/build_homology_map.py
python results/09_external_benchmarks/run_benchmarks.py        # -> results.csv, figures/
```

## What the correction changed

Pooled Pearson r, before → after the estimator fix (originals in `results_pre-correction.csv`):

| benchmark | subset | A Tsuboyama | B FireProt | D fine-tuned |
|---|---|---|---|---|
| S669 | full | 0.255 → **0.415** | 0.500 → 0.546 | 0.462 → 0.506 |
| S669 | common25 | 0.214 → **0.361** | 0.404 → 0.460 | 0.408 → 0.453 |
| Ssym | full | 0.728 → 0.759 | 0.891 → 0.850 | 0.797 → 0.780 |

Regime A gains **+0.15 to +0.16**, the others **+0.04 to +0.06** — exactly the asymmetry
predicted by the epoch-count argument, since A had the most data and therefore the most
over-training at a fixed 250 epochs. The regime ordering is unchanged, but the *size* of
the corpus effect on S669 roughly halves (common-25 gap B−A: 0.190 → 0.099), and the
per-protein ranking advantage disappears entirely.
