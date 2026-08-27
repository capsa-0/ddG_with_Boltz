# 11 — The calibration gap: is there a protein-level term we can predict?

**What:** Why the blind-benchmark *pooled* correlation (S669, homology-filtered
r ≈ 0.45) is so much lower than the *per-protein* correlation (median 0.54–0.59) for
the same predictions — and whether the missing piece can be supplied.

**Why:** results/09 showed the model ranks mutations well inside a protein but
calibrates poorly across proteins. If a single number per protein closes that gap, and
if that number is predictable from the wild-type alone, it is the cheapest large win
available to the project.

**How:** Three questions, in order.
1. *What kind of correction is missing?* Apply the best possible per-protein
   **offset**, **gain**, and **affine** correction to existing predictions and re-score.
2. *Is the gain real?* Estimate the offset from **half** each protein's variants and
   score on the **other half** — separating transferable signal from noise-fitting.
3. *Can it be predicted?* From the Boltz WT embedding, from a whole-protein pooled
   single representation, from interpretable descriptors, and — as the physical
   candidate — via absolute **ΔG(WT)**.

## Headline numbers

| Question | Answer |
|---|---|
| Which correction is missing? | **Offset**, not gain. On the leakage-clean subset the oracle offset lifts regime D from 0.453 → **0.643**, while the oracle *gain* reaches only 0.472 — and gain makes RMSE **worse** in every regime (D: 1.618 → 1.722) even where it nudges r up. Affine adds nothing over offset alone. |
| Is the gain real on S669? | **Yes, though smaller than first reported.** Honest split-half: 0.511 → **0.655** (**+0.144** transferable; +0.055 was noise-fitting). Split-half reliability of the per-protein mean ΔΔG is 0.823, so the offset is a stable quantity, not curation noise. |
| Does it hold in-distribution? | **No — this is the key result.** On held-out Tsuboyama the same ceiling is **+0.029** (0.779 → 0.808), and the offset is 5× smaller (sd **0.29** vs 1.46 kcal/mol). The model is already calibrated in-distribution: the correction is worth **~5× more across datasets than within one**. |
| Predictable from the WT embedding? | **No, on either dataset.** S669: the best head reaches offset r=+0.262 (ridge, regime B) but **applying it never helps** — B 0.546 → 0.542, D 0.506 → 0.490, and a LOPO head drives D to 0.343. Tsuboyama: whole-protein `s` r=+0.038, pooled 0.776 → 0.775. |
| Predictable from structure/chemistry? | **No.** Length, composition, burial and hydropathy are all at or worse than a constant baseline. |
| Predictable via ΔG(WT)? | **No.** Whole-protein Boltz `s` predicts ΔG(WT) at r=**0.293**, against **0.266 from protein length alone**. |
| Do homologues share the offset? | **No** (pair r = 0.090 ± 0.242 for constructs of the same base structure) — even though they *do* share the per-protein **mean ΔΔG** (pair r = **0.516**). |

**Conclusion — the offset is a domain-shift term, not a protein property.**

The decisive comparison is the last row. The per-protein mean ΔΔG *is* a fold property,
shared between close homologues at r = 0.52. The model's **error** on that quantity is
not shared at all. So the model already extracts the part of the protein-level signal
that is determined by the fold; what remains is corpus/assay context, which no
representation *of the protein* can supply — and which is why the correction is worth
+0.144 across datasets and +0.029 within one.

This closes the per-protein-correction direction, and it reframes the S669 deficit:
it is **cross-dataset calibration under domain shift**, not a missing term the model
could have learned. The defensible headline claim for this predictor is **within-protein
ranking** (per-protein median r 0.54–0.59 on S669; ρ 0.789 in-distribution).

Secondary finding: **ΔG(WT) is not recoverable from a frozen Boltz trunk** (r 0.29 vs
0.27 from length alone), and is itself only weakly a fold property (pair r ≈ 0.32,
mean |Δ| 0.64 kcal/mol between constructs differing by one background mutation — about
one mutation's worth of stability). Pooling a representation over residues averages away
exactly what determines it.

## Data & provenance

| Item | Path |
|---|---|
| Benchmark features (S669) | `data/processed/s669/features_ablation.parquet` |
| Training features | `data/processed/tsuboyama_bench_fast/features_ablation.parquet`, `data/processed/fireprot_le500/features_ablation.parquet` |
| Slim store (WT `s`, `pdrow`) | `data/processed/tsuboyama_bench_fast/slim/shard_0000.npz` |
| Homology clusters | `data/processed/tsuboyama_bench_fast/cluster_map_{30,50,90}.csv` — **not** `data/raw/tsuboyama_bench_clusters.csv`, which is degenerate (all 412 proteins in one cluster) |
| S669 leakage map | `results/09_external_benchmarks/homology/s669_leakage.csv` |
| Absolute ΔG(WT) source | `ddg_datasets/dms/Processed_K50_dG_datasets/Processed_K50_dG_datasets/Tsuboyama2023_Dataset2_Dataset3_20230416.csv` (`dG_ML`, `mut_type=='wt'`, join on `WT_name`) |
| Model / regimes | same protocol as `results/09_external_benchmarks/run_benchmarks.py` (concat `wtz`+`mtz`, antisymmetry augmentation, 5-seed MLP `(256,128,64)`) |
| Result tables | `offset_ceiling.csv`, `split_half.csv`, `split_half_tsuboyama.csv`, `homology_share.csv` |
| Intermediates | `data/processed/_analysis/` (gitignored) |

## Code

| Script | What it answers |
|---|---|
| `offset_ceiling.py` | oracle offset / gain / affine ceiling on S669 |
| `offset_real.py` | honest split-half ceiling + interpretable-descriptor prediction |
| `offset_learn.py` | offset head trained on all 550 Tsuboyama+FireProt proteins; writes figure `01` |
| `make_figures.py` | figure `02` — the ceiling contrast and the homology control |
| `build_report.py` | `report.pdf` from the committed tables |
| `exp4_dG.py` | can whole-protein Boltz `s` predict absolute ΔG(WT)? |
| `exp1_offset.py` | offset ceiling + predictability on held-out Tsuboyama |
| `homology_share.py` | do similar proteins share ΔG(WT) / the offset / the mean ΔΔG? |

Run order: `offset_ceiling` → `offset_learn` → `offset_real` → `exp4_dG` →
(`results/12_error_anatomy/tsu_class_error.py`) → `exp1_offset` → `homology_share`.

## Figures & report

- `figures/01_per_protein_error.png` — per-protein signed error and MAE by regime.
- `figures/02_ceiling_and_sharing.png` — the panel that carries the conclusion: the
  ceiling contrast, and what homologues do and do not share.
- [`report.pdf`](report.pdf), regenerated by `build_report.py` from the committed tables.

See [`figures/README.md`](figures/README.md).

## Pending

- `details.md` (methods appendix) is still absent; the README's Data & provenance table
  and this file cover its ground for now.
