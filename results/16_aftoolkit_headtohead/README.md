# 16 — Head-to-head against AFToolkit on S669 and FireProt, with the leakage audit

**What:** a direct comparison between this project's best *transfer* configuration and
[AFToolkit](https://github.com/AIRI-Institute/AFToolkit) (Sindeeva et al., *Brief.
Bioinform.* 26(4) 2025, bbaf324) — the closest published method, and the only one that,
like this project, reads ΔΔG out of a frozen structure model's internal representations
by running both the wild type and the mutant.

**Why it is not a table of published numbers.** AFToolkit releases its precomputed AF2
features for all 669 S669 variants *and* its three trained adapters, so its **own
per-variant predictions** were reproduced here (SVM ρ 0.515 / r 0.518 / RMSE 1.414
against the paper's 0.51 / — / 1.41). That makes S669 a **paired** comparison on the
identical 541 variants this project covers, instead of a comparison of two aggregates
computed on different variant sets. It also lets the leakage question be answered with
AFToolkit's own released training manifest rather than by inference.

## Headline

### S669 — the same 541 variants / 62 proteins, both methods blind

| method | representation | training set | ρ | r | RMSE |
|---|---|---|---|---|---|
| **this project — `zdiag`, 128 d** | Boltz-2 pair-track diagonal, frozen | 12,359 Tsuboyama | **0.569** | **0.557** | **1.357** |
| this project — `diag`+contact-weighted, 256 d | Boltz-2 pair track, frozen | 12,359 Tsuboyama | 0.565 | 0.547 | 1.365 |
| this project — `dz`, 256 d | Boltz-2 pair track, frozen | 12,359 Tsuboyama | 0.552 | 0.541 | 1.425 |
| **AFToolkit (SVM)** | AF2 pair + LDDT logits + pLDDT, frozen | 223,611 cDNA+PROSTATA | 0.511 | 0.525 | 1.401 |
| this project — `concat`, 256 d *(project default)* | Boltz-2 pair track, frozen | 12,359 Tsuboyama | 0.371 | 0.381 | 1.543 |
| AFToolkit (MLP) | same features | same | 0.344 | 0.370 | 1.631 |
| AFToolkit (CatBoost) | same features | same | 0.330 | 0.269 | 1.712 |

Paired protein-cluster bootstrap (4,000 resamples), this project − AFToolkit SVM:

| configuration | Δρ | Δr |
|---|---|---|
| `zdiag` 128 d | **+0.063 [−0.002, +0.144]**, P(ahead) 0.97 | +0.042 [−0.029, +0.133], P 0.84 |
| `dz` 256 d | +0.047 [−0.028, +0.136], P 0.88 | +0.027 [−0.057, +0.132], P 0.71 |
| `concat` 256 d *(default)* | **−0.130 [−0.226, −0.017]** | **−0.132 [−0.220, −0.028]** |

**Read it as parity, not a win.** `zdiag` is nominally ahead on every metric and the
Spearman difference is marginal (95 % CI touches zero at −0.002), but it is not
significant. What *is* significant is the other direction: the project's **adopted
default** loses to AFToolkit by 0.13. The parity is bought entirely by the
transfer-facing readout results/14 recommended — the pair-track **diagonal**, 128
dimensions, against AFToolkit's 358 — and by a training corpus **18× smaller**
(12,359 mutations vs 223,611 samples), both models frozen.

### The 500-residue cap does not flatter this project — measured, not assumed

results/14 could only bound this with a length-independent proxy. With AFToolkit's
per-variant predictions the control is direct: AFToolkit scores **ρ 0.552 on the 128
S669 variants our cap excludes** and **0.511 on the 541 we keep**. The variants we score
are the *harder* half, so the comparison is conservative.

### FireProt — AFToolkit publishes no number, and most of the corpus is in its training set

| test corpus | vs | proteins already seen in training |
|---|---|---|
| S669 (62) | this project's Tsuboyama corpus | **0** at 25 % *and* 30 % MMseqs2 identity |
| S669 (94) | AFToolkit's cDNA+PROSTATA | filtered by its authors at BLAST >36 % identity, e<0.05 |
| FireProt ≤500 (138) | this project's Tsuboyama corpus | **8** at 30 % → scored on the remaining 130 |
| FireProt ≤500 (130) | AFToolkit's cDNA+PROSTATA | **90**, by PDB identity |

Both methods are genuinely blind on S669, and the threshold asymmetry runs *against*
this project (25/30 % excludes more distant relatives than 36 %). FireProt is a
different story: AFToolkit's 2,375 PROSTATA training rows sit on 172 PDB entries drawn
from the same ProTherm/VariBench lineage FireProt is curated from, and **319 (protein,
mutation) pairs are literally identical** to test variants. The cDNA/Megascale half
contributes **zero** overlap. Only **40 proteins / 1,265 variants** are blind to both
methods, and that is the only subset on which a FireProt comparison means anything.

This project on FireProt, for reference (Tsuboyama-trained, homology-filtered):

| configuration | all 130 proteins (n=3,102) | blind to both, 40 proteins (n=1,265) |
|---|---|---|
| `zdiag` 128 d | ρ 0.657 / r 0.645 | ρ 0.635 / r 0.583 |
| `diag`+contact-weighted 256 d | ρ 0.687 / r 0.663 | ρ 0.662 / r 0.593 |
| `dz` 256 d | ρ 0.659 / r 0.647 | ρ 0.641 / r 0.592 |

**AFToolkit's own FireProt predictions are being generated** (see `status.md`); this
section will be completed with the paired comparison on the 40 blind proteins.

## Method

- **This project's predictions** are the per-variant transfer dumps from results/14:
  Tsuboyama-only training (12,359 mutations / 412 proteins), `GroupKFold(5)`, 5-seed MLP
  ensemble, fold models averaged onto the blind corpus, **no antisymmetry augmentation**
  so every configuration is compared under one protocol.
- **AFToolkit's S669 predictions** come from its released `s669_pkls` (AF2 features,
  model_2_ptm, MSA discarded and template masked after the first recycle, 4 cycles,
  mutation-position aggregation) fed to its released `trained_svm/mlp/catboost` adapters.
  Reproducing the paper's headline exactly is the check that this path is faithful.
- **Mapping the two S669 tables.** AFToolkit indexes S669 by PDB entry in PDB numbering;
  this project indexes it by UniProt accession in UniProt numbering, with the opposite
  ΔΔG sign. Neither row order nor residue numbers agree, so the two 669-row tables are
  matched one-to-one on (wild-type aa, mutant aa, ΔΔG) with a Hungarian assignment that
  prefers the PDB entry a UniProt's variants vote for; all 669 match with exact ΔΔG
  agreement, and every assignment lands inside its protein's candidate entries.
- **Homology.** S669 leakage from the MMseqs2 map built in results/09 (WT sequences of
  Tsuboyama ∪ FireProt ∪ benchmark, 80 % coverage, 25 % and 30 % identity); FireProt
  leakage re-derived here from `results/08_finetune_fireprot/splits/cluster_map_30.csv`.
  AFToolkit's overlap is read off its own released training manifest.
- **Uncertainty.** Every difference carries a bootstrap that resamples whole **proteins**,
  because variants within a protein are not independent.

## Data & provenance

| Item | Path |
|---|---|
| Our S669 predictions | `data/processed/_analysis/exp14_s669_results_{s669_locality,onehot_s669,s669_base}.csv` |
| Our FireProt predictions | `data/processed/_analysis/exp14_fpfilt_results_{locality_paired,onehot_fp,farctrl,fact_noaug}.csv` |
| S669 corpus | `data/processed/s669/` (541/62; `data/raw/s669_full669.csv` holds all 669) |
| FireProt corpus | `data/processed/fireprot_le200/` + `fireprot_201to500/` (→ 3,205; 3,102 after filtering) |
| S669 homology map | `results/09_external_benchmarks/homology/s669_leakage.csv` |
| FireProt homology map | `results/08_finetune_fireprot/splits/cluster_map_30.csv` |
| AFToolkit assets | object store (URLs in `run_aftoolkit_s669.py`); cached under `$AFT` |
| AFToolkit cluster install | `/grupos/Marce/estructural/ddG_with_Boltz/aftoolkit` (cranex) |
| Training corpus | `data/processed/tsuboyama_bench_fast/` (12,359 / 412) |

## Code

| Script | What it does |
|---|---|
| `run_aftoolkit_s669.py` | runs AFToolkit's released adapters on its released S669 features and maps them onto our variant ids → `aftoolkit_s669_predictions.csv` |
| `compare.py` | our metrics on both corpora, protein bootstrap CIs, and the two leakage audits → `results_ours.csv`, `bootstrap_ci.csv`, `fireprot_aftoolkit_train_overlap.csv` |
| `headtohead.py` | the paired S669 comparison → `headtohead_s669.csv`, `headtohead_s669_bootstrap.csv` |
| `aft_fireprot_run.py`, `aft_array.sbatch` | (cluster) AF2 feature extraction for FireProt variants |
| `make_figures.py` | `figures/01_s669_headtohead.png`, `figures/02_leakage_audit.png` |

## Figures

- `figures/01_s669_headtohead.png` — both methods on the identical 541 variants, the
  paired bootstrap of the difference, and the cap-is-not-easier control.
- `figures/02_leakage_audit.png` — which benchmark proteins each training corpus has
  already seen, and what restricting FireProt to the blind subset costs this project.
