# Status — 14_biophysical_features

**State:** ✅ Done — conclusions inverted from the first write-up
**Last updated:** 2026-08-27

## Current state

All three items are built, ablated and cluster-bootstrapped, entirely on the **local
workstation** (4 cores / 6 GB RAM, **no GPU** — none of items 1–3 touches one: numpy
over the slim store, sklearn MLP, and HTTP for the alignments). Verdicts: **item 1
adopt** (contact-weighted pooling — significant cross-dataset gain on FireProt ≤500,
138 proteins), **item 2 reject** (neutral; raw chain-size features destroy transfer),
**item 3 reject** (conservation adds exactly nothing over item 1 on deep alignments).

Headline: on FireProt ≤500, `cw` vs `base` gives r **+0.067 [+0.006, +0.126]**,
ρ **+0.079 [+0.015, +0.134]**, MAE **−0.139 [−0.190, −0.072]**, AUC-stab
**+0.057 [+0.016, +0.086]**, all significant. S669 (62 proteins) agrees in sign and
magnitude but is underpowered.

612 WT MSAs were refetched and are **cached under each corpus's `msas/`**, keyed by
`wt_id` — future experiments on these corpora need not touch the server again.

## Next steps

**On making the benchmarks externally comparable.** The 500 aa cap excludes **128 S669
variants / 25 proteins** and **457 FireProt variants / 24 proteins** (Ssym loses none —
all 13 of its proteins are 58–164 aa). Those long chains **do not fit in the cluster GPUs'
VRAM**, so extending the benchmarks by brute force is not available. Options, in
recommended order:

0. **DONE (2026-08-27) — the cap is measured, not a worry.** Attempted the baseline
   re-scoring below; **neither canonical source distributes per-variant predictions**
   (DDGemb hosts datasets only: S669/S2450/ptmul-NR; ThermoMPNN's repo has scripts and
   homology maps, no prediction outputs). Substituted a direct test instead: the excluded
   128 S669 variants are label-matched (ddG −0.97 vs −0.96, sd 1.68 vs 1.63, stabilizing
   25.0 % vs 25.1 %), and a length-independent predictor scores **full 669 r 0.217, our 541
   subset 0.200, excluded 128 r 0.281** — the excluded variants are *easier*, so our subset
   is slightly HARDER than the full benchmark and the comparison is conservative. Bounds
   dataset bias, not method bias (Boltz may fold long chains worse). Full 669 written to
   `data/raw/s669_full669.csv`. **Also: recovering S669 is cheap — 100 of the 128 excluded
   variants are 500–800 aa; only 12 exceed 2,000 aa. The ≤800 band alone reaches 650/669.**
   Useful external artifact spotted: ThermoMPNN ships
   `dataset_splits/s669_vs_megascale_homologues.m8`, an independent S669-vs-Megascale
   homology map that could cross-check results/09's zero-leakage claim.

1. ~~**Don't extend — re-score the baselines on our 541-variant subset.**~~ *(blocked: no
   published per-variant predictions available from either source)* S669's authors
   (Pancotti et al. 2022) benchmark ~20 methods and DDGemb hosts the data
   (`build_datasets.py:32`). Scoring published per-variant predictions on the *same* 541
   variants gives an exact matched comparison — stronger than comparing our subset number
   against published 669-variant aggregates, and it needs **no GPU at all**.
2. **Crop to a local window around the mutation.** Experiment 14's own finding licenses
   this: the transferable signal is local (the diagonal alone matches every 256-d readout;
   whole-chain pooled context *hurts* transfer). Validate before trusting it — re-run
   `fireprot_201to500` proteins cropped and compare against their existing full-length
   features; if cropped ≈ full, apply to the long chains. Crop WT and mutant identically
   and crop the a3m columns to match (a3m columns map 1:1 onto the query).
3. **Memory knobs for the 500–800 aa band** (223 FireProt variants / 15 proteins, plus most
   of S669's missing 128): `max_msa_sequences` is the big lever — the MSA module holds
   ~N_seq x L x c activations, so 1000 → 128 cuts that ~8x, while the pair track
   (L² x 128 ≈ 330 MB fp32 at L=800) is unchanged. `recycling_steps` cuts runtime, not peak
   memory. `--embeddings_only` already avoids the diffusion head.
4. **Excluded regardless:** the **182 variants on 4 proteins >2,000 aa** (max 34,350) — no
   VRAM setting reaches these; report as an explicit exclusion.

- [ ] Promote `zcw` into `ddg/features/build_features.py` behind `feature.blocks`
      (pipeline change, not an experiment — awaiting go-ahead).
- [ ] Re-run the `all` config on the transfer corpora with `n_jobs=1`; it died mid-run
      here (see Log 2026-08-26 afternoon) so it has no transfer CI.
- [ ] Still open project-wide: ranking *within* the stabilizing tail (`stab_rho`) is
      unmoved by every item here, as it was by results/13.

## Blockers

**The README/report headline overstates the result and is being revised.** Two issues,
both found by challenging the baseline rather than by a failed run:

1. **The `base` config is weaker than the project's own prior best.** `results/05`
   transferred Tsuboyama → FireProt ≤500 at **r 0.648 / ρ 0.659 / MAE 0.835** using the
   old **Δz** features, a single MLP on all 12,359 mutations, no antisymmetry. Exp-14's
   `base` (concat + antisymmetry, 5-fold averaged) gets **0.595** on the same test set.
   So `cw`'s 0.672 is **+0.024 over the real prior best**, not +0.067 — and +0.024 is
   inside the bootstrap noise band. The `cw − base` comparison remains internally valid
   as a controlled test *of the pooling change*; it is not evidence that the predictor
   improved.
2. **The S669 numbers do not reconcile with `results/09`.** That study's Tsuboyama-only
   regime scored S669 pooled **r 0.255** (0 leaky proteins, so unfiltered = filtered).
   Exp-14's `base`, also Tsuboyama-only, shows **0.437**. The gap is too large to accept;
   candidate causes are the concat+antisymmetry feature form (results/07 showed it helps)
   or a protocol difference. **Until reconciled, no exp-14 S669 number should be quoted.**

Also note for external comparison: ThermoMPNN transfers to FireProt at ~0.65 and
AFToolkit reports S669 R_p 0.52 — so exp-14 is at **parity**, not ahead, and results/05
was already at parity.

**In flight:** `run_ablation.py --configs dz base cw --no-augment --transfer
fireprot_le500 --out results_headtohead.csv` — Δz vs concat vs contact-weighted under
one identical protocol, which is the comparison that should have been run first.

## Log — newest first

### 2026-08-27 — the headline was wrong; what replicated is the opposite finding

Challenged by the question "is this actually better than what we had?", the whole
experiment was re-run as a controlled design. The original claim did not survive.

**Why the first result was an artifact.** `base` (concat = `wtz`+`mtz`) is not a neutral
reference: it is the *only* configuration with no local term at all, and it is
significantly worse than the project's own prior features (Δz) on transfer
(r −0.138 [−0.234, −0.019] no-aug; −0.055 [−0.099, −0.010] with aug). The reported
"+0.067 for contact weighting" was mostly concat's deficit. `dz` reproduces
results/05 exactly (FireProt r 0.647 vs their 0.648), confirming the protocol is sound.

**Design corrections made.** (i) Antisymmetry augmentation was applied as a half-swap to
*every* block — meaningless for difference-form blocks, so `dz`+augmentation had never
been computable; `augment()` now dispatches on block form and refuses mixed configs.
(ii) A 2×4 factorial {Δz, concat, contact-weighted, +far-shell} × {aug, no-aug} was run
on FireProt so nothing is compared across protocols. (iii) The whole config set was
replicated on S669. (iv) FireProt was homology-filtered for the first time in this
project. (v) A substitution-identity control was added.

**Homology leakage — a gap in the series, not just here.** results/05 reports "zero
`wt_id` overlap" with the training set, which is an *identifier* check. Using the joint
MMseqs2 cluster map that has existed since results/08, **8 of 138 FireProt proteins
share a 30 %-identity cluster with Tsuboyama** (103/3,205 variants, 3.2 %). Every
FireProt transfer number in results/05, results/12 and the first half of this experiment
was computed with that contamination. All conclusions here survive filtering to 130
proteins. **S669 needs no filter** — results/09 established zero Tsuboyama homologues.

**What replicated on both blind corpora (paired cluster bootstrap over proteins):**

| claim | S669 (62 prot.) | FireProt (130 prot., filtered) |
|---|---|---|
| near shell − far shell | **+0.201 [+0.087, +0.315]*** | **+0.069 [+0.010, +0.126]*** |
| diagonal − substitution identity | **+0.263 [+0.167, +0.386]*** | +0.266 (pending CI) |
| diagonal − Δz (256d → 128d) | +0.012 [−0.031, +0.053] | −0.006 [−0.034, +0.024] |
| contact weighting − uniform | **+0.001 [−0.062, +0.058]** | +0.013 [−0.008, +0.037] |

**What did NOT replicate.** Contact weighting. On FireProt it reached significance on
ρ (+0.024*), MAE (−0.057*) and AUC (+0.017*) but never on Pearson; on S669 it is
**exactly zero** (+0.001). Item 1 is therefore **not supported**.

**The qualifier that matters.** `diag` − `dz` in-distribution is
**−0.017 [−0.023, −0.010]*** (412 proteins) — the pooled half genuinely earns ~0.02 r
when train and test share a corpus. "Halve the readout" is a *transfer* statement only.

**Standing conclusions.**
1. Local terms carry the transferable signal; the diagonal `z[i,i]` alone (128d) matches
   every 256-d construction on transfer, and beats substitution identity by +0.26 r.
2. Uniform whole-chain pooling behaves like a far-shell readout (S669: concat 0.476,
   far-shell 0.345 vs diagonal 0.557). Pooled **levels** import corpus-specific context
   and damage transfer; pooled **differences** cancel it and are neutral.
3. All three biology-motivated additions fail: contact weighting (does not replicate),
   burial+biophysics (neutral at best, harmful with raw scale features), MSA
   conservation (zero over `cw` on 100 %-coverage, median-depth-9,474 alignments).
4. In-distribution skill does not predict transfer: concat is the *best* in-distribution
   configuration (r 0.801) and the *worst* on transfer (0.476).

**Retired metric:** `detpr30` gave +0.099, +0.076 and −0.194 across evaluation sets —
opposite signs, never significant. Computed over 30 predictions; it supports no claim.

**Process note.** Two orchestration bugs cost time and are recorded so they are not
repeated: a `pgrep -f "…--configs dz base cw"` wait pattern **prefix-matched**
`--configs dz base cw cw+far`, so a chained job never fired; and a run spawned *inside*
a `Monitor` command was killed with the monitor. Chain on explicit file markers, and
launch work standalone.


### 2026-08-26 — packaged as a result folder

Consolidated `results_all.csv` (every config × evaluation set), regenerated all four
bootstrap tables with the reference config in the filename (an earlier `--ref cw` run
had silently overwritten its `--ref base` counterpart), wrote `README.md`,
`details.md`, `make_figures.py` (2 figures) and `build_report.py` → `report.pdf`.

Figure palette `#1F6FB4/#D95F02/#1B9E77/#7570B3` was chosen by running the CVD
validator: the repo's existing gray-green/teal pair (`#4F5D5A`/`#0E6C68`) **fails**
adjacent-pair separation at ΔE 2.3 (deutan) / 6.4 (normal vision).

**New finding while building figure 1C — the gain is not uniform across proteins.**
Splitting FireProt's 138 proteins at the median baseline MAE: the hard half improves
(mean ΔMAE **−0.125**, 52/69 proteins) and the easy half degrades (**+0.121**, only
23/69). Unweighted across proteins the mean change is **−0.002**; the pooled −0.149
comes from variant-weighting plus the hard half. Recorded in README and report as a
stated caveat — it is the expected signature of a variance-reduction / robustness
mechanism rather than an information gain.

### 2026-08-26 (evening) — item 3 closed: conservation is redundant

FireProt ≤500 was the best substrate item 3 could get: **100 % alignment coverage,
median depth 9,474** (all natural proteins), 138 clusters. 138 MSAs fetched, 0 failures.

Paired bootstrap **vs `cw`** (the decisive comparison): `cw+cons − cw` is
r −0.000 [−0.014, +0.021], ρ −0.006 [−0.017, +0.010], MAE +0.011 [−0.015, +0.035],
AUC −0.013 [−0.025, +0.002] — every CI contains zero — while stabilizing bias is
significantly **worse** (+0.140 [+0.082, +0.182]). `cons` alone is significantly worse
than base on everything.

Interpretation: Boltz already extracts the evolutionary signal from the MSA it is given
(results/04 measured that at +0.08–0.10 r), so re-deriving conservation from the same
alignment is redundant. The deficiency was never missing evolutionary input — it was
how the pair track was pooled.

### 2026-08-26 (afternoon) — FireProt transfer: item 1 reaches significance

The S669 result was **underpowered, not wrong**. `fireprot_le500` turned out to be
exactly the union of `fireprot_le200` (1,543 / 85) and `fireprot_201to500` (1,662 / 53),
and **both keep their slim stores locally** — so the `cw`/`bio` blocks took ~1 min to
build, no GPU, no Boltz re-run. Merged tables written to
`fireprot_le500/features_{bio,msa}.parquet`.

FireProt stores ddG with the **same** sign convention as Tsuboyama (68 % positive),
unlike S669 — the runner now takes `--transfer` with a per-corpus `flip` flag
(`TRANSFER` registry) instead of hardcoding S669.

Point estimates: base r 0.595 / ρ 0.602 → **cw r 0.672 / ρ 0.695**. Bootstrap over the
138 proteins: r +0.067*, ρ +0.079*, MAE −0.139*, AUC +0.057*, stab_bias −0.099*.
Point estimates agree closely with S669's (r +0.067 vs +0.077; MAE −0.139 vs −0.148);
the difference is only that 138 clusters can resolve it and 62 cannot.

**Incident:** the `all` configuration died during fold 4 of the `results_transfer.csv`
run with no Python traceback — consistent with the OOM killer on this 6 GB box
(562 dims × 2 × 12,359 rows after augmentation, 5 MLP members). Not confirmed from
kernel logs (`dmesg` unreadable here). `base`, `cw` and `cw+cons` had completed and
their predictions were saved, so the headline test was unaffected.

### 2026-08-26 (morning) — cluster bootstrap; two reported numbers retracted

`bootstrap.py`: cluster bootstrap over proteins, 400 resamples, paired difference vs a
reference config on the same resample. Runs on saved predictions, no refitting —
seconds locally (an earlier note in this file called it the heavy step; that was wrong).

**Retractions.** (i) The `detpr30` 0.733 → 0.867 result reported for `cw+cons` is **not
significant** (+0.099 [−0.068, +0.267]); it is computed from 30 predictions. Across the
three evaluation sets it later gave +0.099, +0.076 and **−0.194** — opposite signs,
never significant. **`detpr30` should be dropped as a metric.** (ii) The S669 Pearson
gains (+0.082 / +0.095) are point estimates whose CIs straddle zero: +0.077
[−0.034, +0.186] and +0.091 [−0.029, +0.206]. On S669 only **MAE** was significant
(−0.148 [−0.250, −0.011]).

In-distribution (412 proteins), `cw` alone is marginally *worse* — ρ −0.010
[−0.015, −0.005], MAE +0.012 [+0.007, +0.017] — its one real gain being reduced
stabilizing bias (−0.050 [−0.076, −0.025]). Only the combined `all` block shows genuine
in-distribution improvement (ρ +0.014*, MAE −0.010*, AUC +0.010*).

Diagnosis of the S669 weakness: 62 protein clusters, of which **5 proteins hold 51 %**
of the 541 variants (median protein contributes 3). Dropping a single protein moves the
r gain anywhere in [+0.053, +0.122].

### 2026-08-25/26 — items 1, 2 and 3 built and ablated

Plans written to `notes/plans/` (gitignored) before running. Sanity gate passed first:
`base` reproduces results/07's concat+symmetry number exactly (**r = 0.799**).

| Step | Cost |
|---|---|
| `build_bio_features.py` (tsuboyama + s669) | ~3 min |
| `run_ablation.py`, 8 configs | ~50 min (~6 min/config, 5 folds × 5-seed MLP, `n_jobs=2`) |
| `run_ablation.py`, `bio_t` follow-up | ~20 min |
| `fetch_msas.py` (474 WT sequences) | ~45 min, network-bound, 0 failures |
| `build_msa_features.py` | ~4 min (<1 s/protein) |
| `run_ablation.py`, 6 `cons` configs | ~55 min |

**Item 1 point estimates:** S669 r 0.437 → 0.519 at identical dimensionality.
Mechanism control: `cw+far` (far-shell complement, matched 512 dims) gives 0.482 —
below both `base+cw` (0.514) and `cw` alone. `wtcw` is near-orthogonal to `wtz`
(mean |r| = 0.06 per dimension), so it is new information, not a rescaling.

**Item 2:** biophysics alone r 0.494 in-distribution but **0.014 on S669**; added to the
embedding it *cost* transfer. Cause found in the feature distributions — `site_len` is
32–72 aa in Tsuboyama but 50–493 in S669, `site_termdist` 0–35 vs 0–224, so the model
extrapolates far outside its training range. The `bio_t` follow-up (site block reduced
to the dimensionless `site_cn_z` + `site_relpos`) stops the damage (S669 0.425 → 0.524)
but adds nothing over `cw` alone.

**Item 3 setup:** alignment depth splits along natural vs de novo — natural proteins
median depth **8,823**, designed **2**, with 145/156 designed proteins below depth 10.
`msa_has_msa` was re-thresholded at depth ≥ 10 (the original `depth > 1` was constant
at 1.0, i.e. a dead feature). Tsuboyama alignment coverage: **63 %**.

**Deviations from the plan, recorded:** S669 carries 17 variants twice (repeat
measurements, same wt_id+mutation, different ddG) whose *features* are byte-identical —
joined many-to-one after verifying that. The OOF dump was initially written to a fixed
filename and one run clobbered another's predictions; it is now named after `--out`,
and S669/FireProt predictions are persisted too (they were not, initially, which is why
the transfer bootstrap needed a re-run).
