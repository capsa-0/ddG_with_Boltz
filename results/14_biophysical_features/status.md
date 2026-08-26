# 14 — Biology-informed features: status log

## 2026-08-25 — items 1 & 2 built and ablated locally; item 3 not started

**Where it ran:** entirely on the **local workstation**
(`/media/capsa/Programas/ddG_with_Boltz`, 4 cores / 6 GB RAM, **no GPU** —
`nvidia-smi` absent, `torch.cuda.is_available()` False). No cluster job was used and
none is needed: nothing in items 1–3 touches a GPU (numpy over the slim store +
sklearn MLP + HTTP). Plans in `notes/plans/` (gitignored).

### What ran

| Step | Command | Cost |
|---|---|---|
| Build blocks | `python results/14_biophysical_features/build_bio_features.py` | s669 4 s; tsuboyama ~3 min |
| Ablation (8 configs) | `run_ablation.py --configs base cw base+cw cw+far base+bio base+bio_nox base+cw+bio bio` | ~50 min total (~6 min/config, 5 folds × 5-seed MLP, `n_jobs=2`) |
| Follow-up (2 configs) | `run_ablation.py --configs cw+bio_t base+cw+bio_t --out results_bio_t.csv` | running |

Outputs: `data/processed/{tsuboyama_bench_fast,s669}/features_bio.parquet`
(12,359 × 556 and 541 × 556), `results.csv`, `data/processed/_analysis/exp14_oof.csv`,
log `data/processed/_analysis/exp14.log`.

**Protocol** — identical to results/06/07/13 so numbers join that series: corpus
`tsuboyama_bench_fast` (12,359 / 412), GroupKFold(5) on `wt_id`, `make_model("mlp")`
5-seed ensemble, antisymmetry augmentation on train folds, S669 transfer by averaging
the 5 fold models (S669 `ddg` negated per results/13). Sanity gate passed: `base`
reproduces results/07's concat+symmetry number exactly (**r = 0.799** vs 0.799).

### Results

Held-out Tsuboyama (protein holdout) / S669 transfer:

| config | dims | OOF r | OOF ρ | stab ρ | stab bias | detpr30 | ndcg30 | **S669 r** | **S669 ρ** |
|---|---|---|---|---|---|---|---|---|---|
| base (`wtz`+`mtz`) | 256 | 0.799 | 0.776 | 0.268 | +0.613 | 0.733 | 0.468 | 0.437 | 0.474 |
| **cw** (contact-weighted) | 256 | 0.795 | 0.766 | 0.245 | +0.563 | 0.767 | 0.527 | **0.519** | **0.531** |
| base+cw | 512 | 0.799 | 0.782 | 0.243 | +0.575 | **0.900** | 0.547 | 0.514 | 0.544 |
| cw+far (capacity control) | 512 | 0.798 | 0.778 | 0.236 | +0.595 | 0.767 | 0.439 | 0.482 | 0.513 |
| base+bio | 297 | 0.787 | 0.782 | 0.287 | +0.572 | 0.733 | 0.519 | 0.425 | 0.469 |
| base+bio_nox | 285 | 0.805 | 0.785 | 0.260 | +0.577 | 0.700 | 0.506 | 0.396 | 0.389 |
| base+cw+bio | 553 | **0.808** | **0.786** | 0.268 | +0.572 | 0.733 | **0.584** | 0.484 | 0.515 |
| bio (biophysics alone) | 41 | 0.494 | 0.400 | 0.065 | +0.697 | 0.233 | 0.109 | 0.014 | 0.067 |

**Item 1 (contact-weighted pooling) is the win, and it is a transfer win.**
S669 r **0.437 → 0.519 (+0.082)**, ρ 0.474 → 0.531, at *identical* dimensionality —
about the same size as the entire MSA contribution measured in results/04 (+0.08–0.10).
In-distribution it is flat (0.799 → 0.795), exactly as pre-registered: results/03
showed pooled r is data-saturated, so this experiment was always going to be decided
on transfer and on the tail.

Mechanism confirmed by the **`cw+far` control**: at the same 512 dims, replacing the
uniform block with the far-shell complement gives S669 0.482 — *below* `base+cw`
(0.514) and below `cw` alone (0.519). The signal is in the first shell, not in extra
capacity. `wtcw` is also near-orthogonal to `wtz` (mean |r| = 0.06 per dimension), so
it is genuinely new information, not a rescaling.

Tail detection also moves, which results/13 could not achieve by loss reweighting:
**detpr30 0.733 → 0.900** and **ndcg30 0.468 → 0.584** (base → base+cw / base+cw+bio),
while `stab_rho` stays flat — i.e. better *retrieval* of stabilizing mutations without
better within-tail ranking.

**Item 2 (burial + biophysics) is a negative, with a diagnosed cause.**
Biophysics alone reaches only r = 0.494 in-distribution and **r = 0.014 on S669**.
Added to the embedding it *costs* transfer (0.437 → 0.425, and 0.396 without the
interaction terms). Cause identified from the feature distributions: the size-dependent
site scalars are domain-shifted — `site_len` is 32–72 aa in Tsuboyama (mean 53) but
50–493 in S669 (mean 281), and `site_termdist` 0–35 vs 0–224. The model extrapolates
outside its training range on exactly the features results/11 already warned are
protein-level rather than site-level.

**Follow-up `bio_t` settles it** (site block reduced to `site_cn_z` + `site_relpos`,
both dimensionless; `results_bio_t.csv`):

| config | dims | OOF r | OOF ρ | S669 r | S669 ρ |
|---|---|---|---|---|---|
| cw (reference) | 256 | 0.795 | 0.766 | 0.519 | 0.531 |
| **cw+bio_t** | 292 | 0.798 | 0.772 | **0.524** | **0.542** |
| base+cw+bio_t | 548 | 0.793 | 0.784 | 0.511 | 0.537 |

Once the size-dependent scalars are removed the biophysics block stops *hurting*
(S669 0.425 → 0.524) but adds essentially nothing over `cw` alone (0.519 → 0.524,
within noise). **Verdict for item 2: neutral.** The Boltz pair track already encodes
what those 40 numbers carry — consistent with results/12's finding that relative error
is flat across burial tertiles. The transferable lesson is the negative one: never feed
this model raw chain-size features when the target corpus has a different size
distribution.

### Item 3 (MSA conservation) — running

The alignments were deleted with the corpora (only the `no_msa` single-sequence
placeholders remained), so they are being refetched for the **WT sequences only**
(412 Tsuboyama + 62 S669) from the public ColabFold MMseqs2 server —
`fetch_msas.py`, batches of 20, resumable per sequence. Started 2026-08-25 23:39;
log `data/processed/_analysis/exp14_msa.log`. First batch returned real, deep
alignments (e.g. `1fh5H` 15,075 sequences; header renamed to `wt_id` per the pipeline
convention), so the server is healthy and the format is right.

`build_msa_features.py` is written and timed on the first alignments: **<1 s per
protein** (depth capped at 2,000 by *random* subsampling — a3m rows are E-value
ordered, so truncating would keep only the closest homologues and understate column
entropy — then 80 %-identity sequence weighting, per-column weighted profile,
entropy, consensus, PSSM log-odds). Sanity: mean column entropy 1.4–1.8 nats against
a ln(20) = 3.0 ceiling. The `cons` block and four configurations
(`cons`, `base+cons`, `cw+cons`, `all`) are already wired into `run_ablation.py`;
`load()` picks up `features_msa.parquet` only if it exists.

### Next

1. Build `features_msa.parquet` when the fetch completes, run the four `cons`
   configurations.
2. Report the `cons` result **split natural vs de novo** — a gain on designed
   proteins with no alignments would be an artifact, not conservation.
3. Cluster-bootstrap `base` vs `cw` over the 412 proteins (the results/13
   `bootstrap.py` pattern) before the +0.082 is quoted as real. This is the one
   genuinely heavy step — 400 resamples — and is the piece that belongs on the
   cluster (nothing else here needs a GPU at all).
4. If contact weighting holds up: it is a change to
   `ddg/features/build_features.py`, not just an experiment — the `zpool` block would
   gain a `zcw` sibling behind `feature.blocks`.
