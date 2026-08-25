# 10 — Full mutational scan of α-galactosidase A (GLA)

**What:** Predict ΔΔG for **every possible single point mutation** of human
α-galactosidase A — 398 residues × 19 substitutions = **7,562 mutations** — with the
Boltz-2 embedding predictor, and compare against a FoldX scan of the same protein.

**Why:** Every result so far (01–09) scores the predictor on *labelled benchmarks*.
This is the first time it is used the way it would actually be used: pointed at one
protein of interest with no labels at all, asked to rank the whole mutational
landscape. GLA is a good target — it is a real 398 aa human enzyme (Fabry disease),
well outside the small designed domains that dominate the Tsuboyama training corpus,
and an independent FoldX scan already exists for it.

## Protein

| | |
|---|---|
| Protein | human α-galactosidase A, mature chain |
| UniProt | P06280, residues **32–429** |
| Length | 398 aa |
| Numbering | results are reported in **UniProt numbering (32–429)**, matching the FoldX table. The pipeline works 1-based internally (`mutation_local`/`position_local` keep those labels). |

## Method

- **Features:** concat Boltz-2 pooled embeddings `wtz` + `mtz` (128 + 128), the
  representation adopted in [`07`](../07_feature_symmetry_ablation/). Extracted by the
  normal pipeline (prepare → predict → per-shard slim → features) with
  `feature.blocks: [zdiag, zpool, wtz, mtz]`.
- **Model:** 5-seed MLP ensemble `(256,128,64)` with antisymmetry augmentation —
  identical machinery to [`09`](../09_external_benchmarks/).
- **Three training regimes**, all reported per mutation:

  | Regime | Training | S669 pooled r / per-protein median (from 09) |
  |---|---|---|
  | **A** | Tsuboyama only (12,359 muts) | 0.26 / 0.46 |
  | **B** | FireProt ≤500 only (3,205 muts) | 0.50 / 0.56 |
  | **D** | Tsuboyama → FireProt fine-tuned | 0.46 / **0.61** |

  Reporting all three is deliberate: they differ only in *training distribution*, so
  the **spread between them** (`ddg_regime_sd`) marks mutations where the prediction
  is extrapolating past what any corpus covers. `ddg_mean` is their average.
- **Sign convention:** positive ΔΔG = destabilizing (all training corpora agree).
  With no labels there is nothing to auto-detect a flipped convention against.

## Comparison baseline — FoldX

`ddg_varmed_by_mutation_foldx.csv` is an independent FoldX scan of the same protein.
Checked against the sequence: all 7,562 substitutions present exactly once, wild-type
residue correct at all 398 positions, no duplicates. Two caveats:

- **40 substitutions have no value** (99.5 % filled): `E58D`, `V137R`, and all 19 at
  each of `L428` / `L429` (the C-terminal pair).
- FoldX values span **[−5.36, +70.27]** kcal/mol, mean +3.16. The large positive tail
  is steric-clash artifact, not a calibrated ΔΔG — comparisons should use rank
  correlation, or clip, rather than RMSE against it.

The 398 extra rows in that file are wild-type-to-itself (`X→X`) entries, not mutations.

## Data & provenance

| Item | Path |
|---|---|
| Scan module | `ddg/scan/` (`build.py`, `predict.py`, `plots.py`, `mutations.py`) |
| Mutation table | `data/raw/scan_GLA_human.csv` (7,562 rows, 1-based) |
| Experiment config | `experiment_configs/scan_GLA_human.yaml` |
| Processed dir | `data/processed/scan_GLA_human/` (cluster; gitignored) |
| Feature table | `data/processed/scan_GLA_human/features_summary.parquet` |
| Scan output | `data/processed/scan_GLA_human/scan/` |
| Train (A, D) | `data/processed/tsuboyama_bench_fast/features_ablation.parquet` |
| Train (B, D) | `data/processed/fireprot_le500/features_ablation.parquet` |
| FoldX baseline | `ddg_varmed_by_mutation_foldx.csv` (this folder) |

## Reproduce

```bash
# 1. generate the mutation table + experiment config
python -m ddg.scan build --name GLA_human --first-residue 32 --sequence <SEQ>

# 2. extract Boltz features (cluster; 7,563 structures)
./slurm/submit_scan.sh experiment_configs/scan_GLA_human.yaml 128 2

# 3. score every mutation under regimes A/B/D
sbatch slurm/scan_predict.sbatch experiment_configs/scan_GLA_human.yaml
```

## Runs

| Experiment | Scope | State |
|---|---|---|
| `scan_GLA_human_hard` | 38 targeted positions (the 10 flagged + all 31 glycines), 722 mutations | **Scored** — 604/722 have features; results and figures in this folder |
| `scan_GLA_human` | all 398 positions, 7,562 mutations | **Running** on the cluster (array 1031 → slim 1399 → features 1400) |

## Headline so far (targeted subset, 604 mutations)

- Predicted ΔΔG means: **A +0.28**, **B +1.24**, **D +0.91** kcal/mol; regimes agree at
  Pearson 0.63–0.80, mean across-regime SD 0.52.
- **vs FoldX: Spearman +0.504** — but **+0.379** once FoldX's clash tail (>10 kcal/mol)
  is removed, and **+0.312** restricted further to exposed sites. Agreement *falls* as
  FoldX becomes more trustworthy, so most of the headline number is both methods
  agreeing that buried-glycine substitutions are bad.
- **No measured ΔΔG exists for GLA** (ProThermDB / ThermoMutDB / FireProtDB / MaveDB all
  checked; the one biophysical study reports urea C₀.₅, not ΔΔG). GLA is also absent from
  every training corpus, so the predictions are leakage-free — but unvalidatable here.
- **Neither the "glycines are worse" nor the "flagged positions are worse" hypothesis
  separates from noise** at ~15 substitutions per position (p = 0.077 and p = 0.476;
  bootstrap CI for a single position's ρ spans [+0.09, +0.83]).

**The overestimation question is open.** It needs measured ΔΔG — i.e. S669/Ssym
(results/09 corpora), not FoldX on this protein.

## Status

See [`status.md`](status.md) for the full work log. `report.pdf` follows once the
398-position run lands and the comparison is redone without the glycine bias.
