# 14 — Biology-informed features: three additions that failed, and what the controls found

**What:** Three biology-motivated additions to the feature construction that sits on top
of the frozen Boltz-2 trunk, each tested as a feature-block ablation with
cluster-bootstrapped confidence intervals on **two** blind corpora.

1. **Contact-weighted pooling** — replace the uniform mean over the whole chain with a
   weighted mean using Boltz's own predicted distogram as the weight.
2. **Burial + per-residue biophysics** — volume, hydropathy, transfer free energy,
   charge, helix/sheet propensity, Gly/Pro flags, and their interactions with burial.
3. **MSA conservation** — column entropy, Neff, PSSM log-odds, consensus indicator.

**All three fail.** What the experiment actually establishes came from its *controls*.

**Why:** The readout was naive in a specific, checkable way. `build_features.py` reduced
the pair track to `z[i,i]` plus `z[i,:]` **mean-pooled uniformly over every residue**, so
a residue 60 Å away weighed as much as a contacting one — and no biophysical or
evolutionary variable existed anywhere in `ddg/`. The choice is not idiosyncratic:
AFToolkit, the closest published method (AlphaFold2 pair representations + adapter
models), enumerates four aggregations (global mean/sum, mutation-site mean/sum), **none
spatially weighted**; Boltz-2's own affinity head likewise "performs mean pooling over
all pairwise interactions".

**How:** Training is 12,359 Tsuboyama mutations / 412 proteins, `GroupKFold(5)` on
`wt_id`, 5-seed MLP ensemble; the fold models are averaged onto two blind corpora.
**Antisymmetry augmentation is treated as an experimental factor, not a fixed default.**
Every claim carries a **cluster bootstrap over proteins** (400 resamples) reported as a
paired difference on the same resample. FireProt is **homology-filtered** (below).

## Headline

**The transfer ladder** — Pearson r, no augmentation, ordered by construction:

| construction | dims | S669 (62 prot.) | FireProt ≤500 (130 prot.) | in-distribution |
|---|---|---|---|---|
| substitution identity (one-hot) | 40 | 0.295 | 0.381 | 0.505 |
| far-shell pooling | 256 | 0.345 | 0.504 | 0.740 |
| **uniform pooling, levels — the project default** | 256 | **0.381** | **0.476** | **0.801** |
| contact-weighted levels | 256 | 0.487 | 0.657 | 0.793 |
| Δz (diagonal + uniform pooled difference) | 256 | 0.541 | 0.647 | 0.792 |
| diagonal + contact-weighted difference | 256 | 0.547 | **0.665** | 0.791 |
| **diagonal alone** | **128** | **0.557** | 0.647 | 0.775 |

The project's current feature form is **third from the bottom on both blind corpora** —
nearer to far-shell pooling and to a plain amino-acid lookup than to the diagonal — while
being the **best** configuration in-distribution. Replacing it with the diagonal alone
halves the readout and gains, on S669, **r +0.173 [+0.055, +0.285]**.

**Claims that replicated on both corpora** (paired bootstrap, Δ Pearson r):

| claim | S669 (leakage-free) | FireProt (filtered) |
|---|---|---|
| diagonal − substitution identity | **+0.263 [+0.167, +0.386]** | **+0.264 [+0.219, +0.308]** |
| local term − far-shell pooling | **+0.201 [+0.087, +0.315]** (diag) | **+0.069 [+0.010, +0.126]** (cw) |
| diagonal − Δz (256d → 128d) | +0.012 [−0.031, +0.053] | −0.006 [−0.034, +0.024] |
| uniform-pooled levels − Δz | — | **−0.141 [−0.241, −0.020]** |

**The three additions:**

| item | verdict | evidence |
|---|---|---|
| 1 — contact-weighted pooling | **not supported** | FireProt ρ +0.024 [+0.001, +0.045] but r +0.013 [−0.008, +0.037]; **S669 r +0.001 [−0.062, +0.058]** — does not replicate |
| 2 — burial + biophysics | **rejected** | never exceeds the baseline; raw chain-scale features actively destroy transfer |
| 3 — MSA conservation | **rejected** | r +0.001 [−0.014, +0.019] over `cw` on 100 %-coverage, median-depth-9,474 alignments, while worsening stabilizing bias (+0.137 [+0.079, +0.182]) |

## Interpretation

The pair track's **local** term carries the transferable signal, and it is not an
amino-acid lookup — the diagonal beats substitution identity by +0.26 r on both corpora.

Whole-chain **uniform pooling behaves like a far-shell readout** (S669: concat 0.381,
far-shell 0.345, diagonal 0.557), because a uniform mean over a chain is dominated by
distant residues. The decisive distinction is between pooled **levels** and pooled
**differences**: `zpool = mtz − wtz` cancels the per-protein offset and is harmless;
`wtz`/`mtz` retain it, and that offset is corpus-specific — the term results/11 identified
as domain shift rather than a learnable protein property. Adding the levels to the
diagonal *costs* 0.08 r on FireProt.

**This is a train/test-mismatch story, not "these features are useless."** In-distribution
the pooled half genuinely earns its keep: `diag` − `dz` is **−0.017 [−0.023, −0.010]** on
held-out Tsuboyama. The recommendation is therefore conditional on the deployment regime,
and in-distribution holdout performance is actively misleading here — the best
in-distribution configuration is among the worst on transfer.

## Limits on external comparison — read before quoting any number

**Neither blind corpus is the published benchmark.** `results/09_external_benchmarks/build_datasets.py`
caps proteins at 500 residues (`CAP = 500`), so:

| benchmark | published | used here |
|---|---|---|
| S669 | 669 variants / 94 proteins | **541 / 62** (81 %) |
| FireProt ≤500 | full FireProt | length-capped, then homology-filtered to 130 proteins |

Every configuration is scored on the identical subset, so all **internal** comparisons in
this report are valid and the cap cancels. The absolute values are still **not directly
comparable to published S669/FireProt figures** (AFToolkit's S669 R_p 0.52, ThermoMPNN's
~0.65 on FireProt), and **no claim of state-of-the-art performance is made.**

**But the cap does not flatter us — measured, not assumed.** The 128 excluded S669 variants
are label-matched to the kept ones (ddG mean −0.97 vs −0.96, sd 1.68 vs 1.63, stabilizing
25.0 % vs 25.1 %), and a **length-independent** predictor (substitution identity, 40 one-hot
dims, which needs no embeddings and so runs on any chain length) scores them *higher*:

| S669 view | n | r | ρ |
|---|---|---|---|
| full 669 | 669 | +0.217 | +0.244 |
| our subset (≤500) | 541 | **+0.200** | +0.225 |
| excluded (>500) | 128 | **+0.281** | +0.318 |

So our subset is if anything **slightly harder** than the full benchmark (−0.017 r), making
comparison to published 669-variant numbers conservative rather than optimistic. *Caveat:*
this bounds **dataset** bias, not **method** bias — it cannot rule out that Boltz folds long
chains less accurately, which would penalise our model there specifically.

**Recovering the full benchmark is mostly cheap.** Of the 128 excluded S669 variants,
**100 are 500–800 aa** (14 proteins, max 795) and only **12 are >2,000 aa** (4 proteins,
max 3,685). Running the ≤800 aa band alone would reach **650/669 = 97 %** of S669. Testing that would require Boltz
predictions for the >500 aa proteins, which do not exist in this project.

**An unreconciled discrepancy with results/09.** That study's regime A is the same setup
(Tsuboyama-only, concat + antisymmetry, same 541 variants) and reports S669 r = **0.255**;
`base` here gets **0.437**. The estimators differ — results/09 uses
`early_stopping=False, max_iter=250` where `make_model("mlp")` early-stops on a validation
split — which would overfit and disproportionately damage transfer. Until this is settled,
treat exp-14's *absolute* S669 values as provisional; the paired differences are unaffected,
since both arms of every comparison use the same estimator.

## Data & provenance

| Item | Path |
|---|---|
| Training corpus | `data/processed/tsuboyama_bench_fast/` (12,359 / 412) |
| Blind corpus A | `data/processed/s669/` (541 / 62; **ddG sign inverted**; zero Tsuboyama homologues per results/09) |
| Blind corpus B | `data/processed/fireprot_le500/` (3,205 / 138 → **130 after filtering**), assembled from shards `fireprot_le200` + `fireprot_201to500`, which hold the slim stores |
| Homology map | `results/08_finetune_fireprot/splits/cluster_map_30.csv` |
| Embedding source | `slim/*.npz` — `zrow` (pair track), `pdrow` (distogram logits, 64 bins over 2–22 Å) |
| Feature tables | `features_ablation.parquet` (`wtz`,`mtz`,`zdiag`,`zpool`), `features_bio.parquet` (`wtcw`,`mtcw`,`wtfar`,`mtfar`,`cwd`,`site_*`,`oh_*`), `features_msa.parquet` |
| MSAs (refetched) | `data/processed/*/msas/{wt_id}.a3m` — 612 alignments, 0 failures |
| Prediction dumps | `data/processed/_analysis/exp14_{oof,s669,fireprot_le500,fpfilt}_*.csv` |
| Result tables | `results_all.csv` (config × set × protocol), `bootstrap_*_ref-*.csv` |
| Model | `make_model("mlp")` from `ddg/evaluation/models.py`, unchanged |

**Homology filtering — a correction to the series.** results/05 reports "zero `wt_id`
overlap" with the training set; that is an *identifier* check. Using the joint MMseqs2
map that has existed since results/08, **8 of 138 FireProt proteins share a 30 %-identity
cluster with Tsuboyama** (103/3,205 variants, 3.2 %). Every FireProt transfer number in
results/05, results/12 and the first pass of this experiment carried that contamination.
All conclusions here survive filtering; the filtered numbers are primary throughout.

## Code

| Script | What it does |
|---|---|
| `build_bio_features.py` | slim store → contact-weighted / far-shell pooling, burial, biophysics |
| `fetch_msas.py` → `build_msa_features.py` | refetch WT alignments; per-column conservation features |
| `run_ablation.py` | the ablation: `--configs`, `--transfer {s669,fireprot_le500}`, `--no-augment` |
| `bootstrap.py` | cluster bootstrap over proteins, paired vs a reference config |
| `consolidate.py` | merge per-run tables into `results_all.csv`, tracking protocol |
| `class_split.py` | natural vs de novo, and the results/12 deficit classes |
| `make_figures.py`, `build_report.py` | figures and `report.pdf` |

## Figures

- `figures/01_what_generalizes.png` — the transfer ladder on both corpora; the replicated
  claims with paired CIs; the in-distribution/transfer reversal.
- `figures/02_additions_that_failed.png` — the three additions, each with its CI.

## Next

Replace `zpool` with `zdiag` as the transfer-facing default in
`ddg/features/build_features.py` (a `feature.blocks` change), keeping the pooled block
available for in-distribution use. Unmoved by everything here: ranking *within* the
stabilizing tail (results/12's deficit), as it was by results/13's loss reweighting.
