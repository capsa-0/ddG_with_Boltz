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

### S669 — the same 629 variants / 71 proteins, both methods blind

The pipeline's 500-residue cap originally left this at 541 of 669 variants. Measuring the
real hardware ceiling (below) showed the 505–701 aa band was reachable, so `s669_ext`
extracted its 88 variants and the corpus is now **629/669 = 94 %** of the published
benchmark.

| method | representation | training set | ρ | r | RMSE |
|---|---|---|---|---|---|
| **this project — `zdiag`, 128 d** | Boltz-2 pair-track diagonal, frozen | 12,359 Tsuboyama | **0.583** | **0.574** | **1.295** |
| this project — `diag`+contact-weighted, 256 d | Boltz-2 pair track, frozen | 12,359 Tsuboyama | 0.578 | 0.565 | 1.303 |
| this project — `dz`, 256 d | Boltz-2 pair track, frozen | 12,359 Tsuboyama | 0.568 | 0.559 | 1.366 |
| **AFToolkit (SVM)** | AF2 pair + LDDT logits + pLDDT, frozen | 223,611 cDNA+PROSTATA | 0.533 | 0.548 | 1.327 |
| AFToolkit (MLP) | same features | same | 0.381 | 0.400 | 1.558 |
| this project — `concat`, 256 d *(project default)* | Boltz-2 pair track, frozen | 12,359 Tsuboyama | 0.373 | 0.385 | 1.507 |
| AFToolkit (CatBoost) | same features | same | 0.358 | 0.309 | 1.635 |

Paired protein-cluster bootstrap (4,000 resamples), this project − AFToolkit SVM:

| configuration | Δρ | Δr |
|---|---|---|
| `zdiag` 128 d | **+0.054 [−0.003, +0.124]**, P(ahead) 0.97 | +0.033 [−0.029, +0.111] |
| `diag`+cw 256 d | +0.046 [−0.021, +0.120], P 0.91 | +0.023 [−0.053, +0.109] |
| `dz` 256 d | +0.039 [−0.026, +0.120], P 0.86 | — |
| `concat` 256 d *(project default)* | **−0.153 [−0.241, −0.054]** | — |

**The extra 88 variants do not change the verdict — parity, not a win.** `zdiag` is
nominally ahead on every metric and the Spearman interval still just touches zero. Both
methods gain slightly on the larger corpus (AFToolkit 0.511 → 0.533, this project
0.569 → 0.583), so the earlier 541-variant reading was not an artefact of the cap. This
project reaches that with **128 dimensions against 358** and a training corpus **18×
smaller**, both backbones frozen.

What *is* significant runs the other way: the project's **adopted default** (`concat`)
loses to AFToolkit by **−0.153 [−0.241, −0.054]**. The parity above is bought entirely by
the transfer-facing readout results/14 recommended, and would be missed by anyone taking
the pipeline's own default configuration.

<details><summary>The original 541-variant view, for continuity with earlier commits</summary>

| method | ρ | r | RMSE |
|---|---|---|---|
| this project — `zdiag` 128 d | 0.569 | 0.557 | 1.357 |
| AFToolkit (SVM) | 0.511 | 0.525 | 1.401 |
| this project — `concat` 256 d *(project default)* | 0.371 | 0.381 | 1.543 |
| AFToolkit (MLP) | 0.344 | 0.370 | 1.631 |
| AFToolkit (CatBoost) | 0.330 | 0.269 | 1.712 |

Paired Δρ for `concat` was **−0.130 [−0.226, −0.017]** — the project's *adopted default*
loses to AFToolkit significantly. The parity above is bought entirely by the
transfer-facing readout results/14 recommended.
</details>

### The length cap: measured, then lifted

**The ceiling was measured, and it was not 500.** A length-ladder probe on an 8 GB
RTX 2080 (compute capability 7.5, so Boltz's own triangle kernels are force-disabled) put
the real limit between 701 and 795 aa:

| chain (aa) | 505 | 619 | 648 | 701 | 795 | 801 | 1207 |
|---|---|---|---|---|---|---|---|
| peak VRAM (MiB) | 4,991 | 6,395 | 6,769 | 7,465 | 7,711 | 7,755 | 7,195 |
| result | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| GPU time | 2.4 m | 3.6 m | 3.9 m | 4.6 m | — | — | — |

So the 88 variants on 9 proteins in the 505–701 aa band were extracted (~5.3 GPU-hours)
and folded in: **81 % → 94 %** coverage. The remaining 40 sit on proteins ≥724 aa and need
more VRAM or an Ampere card, where the kernels switch on.

**The length bands differ sharply in difficulty — so absolute scores are not comparable
across corpora, and the paired Δ is.** Measured with AFToolkit's predictions, so the
difficulty estimate is independent of the method under test:

| band | n | AFToolkit ρ |
|---|---|---|
| base, ≤500 aa | 541 | 0.511 |
| **added, 505–701 aa** | 88 | **0.761** |
| still excluded, >701 aa | 40 | **0.327** |

The band we added is markedly *easier* than the base corpus and the tail we still cannot
reach is much harder, so extending to 629 raised both methods' absolute numbers
(AFToolkit 0.511 → 0.533, this project 0.569 → 0.583). **This corrects an earlier reading
in this folder**: on the old split the excluded 128 looked slightly *easier* overall
(ρ 0.552 vs 0.511), which was a blend of the easy 88 and the hard 40 — the cap was not
handing us the harder half, it was handing us a mixture.

What matters is that the *paired* difference barely moves — **Δρ +0.063 [−0.002, +0.144]
on 541 against +0.054 [−0.003, +0.124] on 629** — so the parity verdict is robust to which
of the two corpora you score, which is exactly why the comparison is run paired.

**A silent-failure trap, now guarded.** For all three over-long chains Boltz **exited 0
having written nothing**, and the pipeline logged `Feature extraction complete!`. Nothing
in `subprocess.run(..., check=True)` catches that, so a run containing long proteins would
quietly lose them and report success. `run_boltz.py` now compares collected predictions
against pending queries and raises.

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

### FireProt — the leakage reversal

AFToolkit's own FireProt predictions were generated by re-running its published pipeline
(AlphaFold2 `model_2_ptm`, MSA discarded and template masked after the first recycle,
4 cycles, mutation-position aggregation) over 2,983 variants on the cluster, then feeding
its released SVM adapter. **84 variants (2.8 %) fail inside AFToolkit's own code** —
33 `AssertionError`, 28 a bare `raise` with no active exception, 23 `KeyError` on a
residue number — all on structures with unresolved residues. Both methods are scored on
the surviving 2,899, so the comparison stays paired.

Split by whether AFToolkit has already trained on the protein (Spearman ρ):

| subset | variants / proteins | AFToolkit SVM | this project (`diag`+cw) | paired Δ |
|---|---|---|---|---|
| proteins **AFToolkit trained on** | 1,726 / 88 | **0.755** | 0.716 | **−0.063 [−0.164, −0.004]** |
| **blind to both methods** | 1,173 / 36 | 0.633 | **0.685** | +0.046 [−0.019, +0.094] |
| (naive, all scored) | 2,899 / 124 | 0.706 | 0.696 | −0.016 [−0.077, +0.023] |

**The ordering flips.** Where AFToolkit has seen the protein it wins by a statistically
significant margin. Remove those proteins and it loses **−0.122 ρ**, against this
project's **−0.031** — and the sign of the difference reverses. The naive whole-corpus
comparison, which is what anyone reading the two papers would compute, splits the
difference and is wrong about the direction.

The blind-subset difference is not significant, and it sits inside the ~0.015 ρ that our
re-derivation of AFToolkit's features costs it (below), so the honest reading is
**parity on blind proteins** — the same conclusion S669 reaches, now replicated on a
second corpus with an independent backbone.

**Reproduction fidelity.** AFToolkit's released S669 features could be compared against
ones we re-derive from raw PDB on our own hardware: features agree to 1.4 % (mean
absolute difference 1.49 against a mean magnitude of 105), and its SVM's predictions
agree at ρ 0.975 / mean |Δ| 0.08 kcal/mol, costing it about **−0.015 ρ** against
experiment. The FireProt numbers above are therefore a slight *under*estimate of
AFToolkit, and are reported as such.

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
