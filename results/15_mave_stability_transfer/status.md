# Status — 15_mave_stability_transfer

**State:** 🚧 In progress — GPU chain 1414→1415→1416→1417 submitted; Phase 0 harness reproduction running locally.
**Last updated:** 2026-08-26

## Current state

Testing whether our Boltz-embedding ΔΔG carries stability signal competitive with
Rosetta ΔΔG for predicting **MAVE functional fitness** (Høie et al. 2022, Cell Reports
38:110207 — `theory/biblio/marce/RF4Mave.pdf`). Planned in `TODO.md` §4.

Done so far, all CPU-only on the workstation:
- Høie data fetched and verified usable (39 datasets / 29 proteins / 212,450 scored
  single variants; 100 % WT-residue agreement with their header sequences).
- Tier-1 corpus built: **11 proteins ≤200 aa, 13 MAVE datasets, 25,224 Boltz
  structures**, 0 WT mismatches.
- Homology leakage map built: **only UBI4 (ubiquitin) is leaky** vs Tsuboyama.
- Phase-0 harness reproduction of their published LOPO medians: in progress.

## Next steps
- [ ] **After array 1415 finishes: backfill the 6 lost shards** (98 + 132–136,
      ~594 structures, 2.4 % of the corpus) before trusting the feature table.
      The cluster checkout is on `main` and does **not** yet have the nodo4 exclude,
      so land the two updated files first, then re-submit:
      ```bash
      B=origin/results/11-12-calibration-and-error-anatomy
      git fetch origin results/11-12-calibration-and-error-anatomy
      git show $B:slurm/predict_array.sbatch > slurm/predict_array.sbatch
      git show $B:slurm/submit_scan.sh      > slurm/submit_scan.sh
      ./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3
      ```
      (These two are *tracked* on the cluster, unlike the config/CSV landed earlier,
      so this leaves them showing as modified until the branch is merged.)
      predict skips everything already in the slim store, so it redoes only the gap.
      Verify afterwards: slim structure count should reach 25,224.
- [ ] After ~5 predict shards land, measure real s/structure and re-derive the ETA
      before letting the rest run (the failure mode that killed the results/10 full
      scan). Budget is extrapolated from 65 s/structure at 398 aa; the exponent for a
      65–189 aa protein is unverified.
- [x] `check_frames.py` — score.py's feature rebuild verified (max |Δ| 0.042, PASS).
- [ ] `rsync` the slim store (~4.1 GB, `keep_s: true`) back here so different models
      and feature blocks can be tried without the cluster.
- [ ] Phase 3: `predict_ddg.py` (regimes A/B/D) → `score.py` (direct + LOPO layers).

## Blockers
- None.

## Log — newest first

### 2026-08-26 — shard 1415_98 hung on nodo4; cancelled, throughput restored

At the halfway mark (128/256) the rate had dropped from 0.246 to 0.129 shards/min.
Cause: **shard `1415_98` had been RUNNING 4 h 06 m on nodo4** against a normal 13–20 min,
holding one of the three concurrent slots.

It was hung, not slow: its logs stopped at **10:23** and it was then **14:30** — four
hours with no output, progress bar frozen at structure 85/99, ~7:48 into prediction.
No traceback, no CUDA error; it simply stopped. nodo4 is not obviously a bad node — it
ran shard `1415_1` in 9 m 52 s earlier — so this reads as a one-off stall (NFS or GPU
wedge) rather than a node to add to the exclude list. Watching for a repeat.

`scancel 1415_98`. Verified afterwards: shard gone from the queue, slots refilling
(`1415_130`, `1415_131` picked up immediately), and **the chain is intact** — 1416/1417
are still plain `PENDING`, not `DependencyNeverSatisfied`, because `submit_scan.sh`
attaches the slim sweep with `afterany` precisely so one dead shard cannot strand the
run.

**Shard 98's ~99 structures are simply missing** — no raw NPZs were left behind (Boltz
appears to write its outputs at the end of a run, not per structure). Recovery is the
documented one: after the array finishes, re-run
`./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3`, which skips
everything already in the slim store and redoes only the gap. **Do not skip this step** —
without it the corpus is short ~99 of 25,224 structures (0.4 %), which would silently
shrink the feature table rather than fail loudly.

Revised ETA with three slots restored: ~127 shards × ~15 min ÷ 3 ≈ **10.5 h**, so around
01:00 on 2026-08-27, plus a short gap-filling run.

### 2026-08-26 — dry-run of Phase 3 on synthetic predictions; found and fixed a fairness bug

With the GPU run a third done, exercised the Phase-3 path that `check_frames.py` did
not cover, using a synthetic predictions CSV with the exact schema `predict_ddg.py`
emits (values meaningless — only the plumbing under test).

**Layer 1 join verified.** All 13 datasets join cleanly on `(wt_id, mutation)` with the
right row counts, and `rho_rosetta` comes out at median **−0.301** — exactly the direct
Rosetta baseline computed independently from the PRISM tables, per dataset as well as in
aggregate. Sign is negative as it must be (destabilizing → low fitness); the sign guard
stays quiet.

**Found: the two arms were not actually paired.** Our scan is *full saturation*, so it
has a ΔΔG for **100 %** of scored variants and **95.0 %** of position-grid cells (19/20 —
the WT cell has no mutation). Rosetta has **95.7 %** and **90.9 %**; their calculations
have genuine gaps. Two consequences, both of which would have inflated our arm:

1. The position-context model would have given the Boltz arm denser features — winning
   partly on *coverage* rather than on ΔΔG quality.
2. Worse, their `-x 2` filter drops rows whose own stability value is missing, and it
   reads whichever column is in the ΔΔG slot. So the Boltz arm would have been scored on
   ~4.3 % **more rows** than the Rosetta arm — and precisely the rows Rosetta could not
   compute, which are unlikely to be a random sample.

**Fix:** `build_frames(..., match_coverage=True)` (now the default) masks our ΔΔG to
Rosetta's availability pattern, so both arms see identical missingness and identical row
sets. Verified: both arms now report 95.7 % own-value and 90.9 % grid coverage. The
paired difference now isolates ΔΔG *quality*, which is the thing under test.
`--no-match-coverage` measures separately what the full-saturation advantage is worth —
a real benefit of our method, but one that deserves to be reported as its own number
rather than smuggled into the headline.

### 2026-08-26 — feature rebuild verified (check_frames.py): PASS

The other half of the harness. Phase 0 validated the LOPO *using Høie's own feature
tables*; Phase 3 needs `score.py` to rebuild those 47 features from the raw PRISM
tables so our ΔΔG can take Rosetta's place. That rebuild is our code, and a divergence
would corrupt the headline number in a way Phase 0 cannot see.

Same LOPO (60 trees), same 13 Tier-1 datasets, run over both feature sources:

| | median ρ |
|---|---|
| their `preprocessed.pkl` | +0.502 |
| our rebuild from PRISM | +0.510 |

**max \|Δ\| = 0.042, mean \|Δ\| = 0.019**, both feature sets exactly 47 columns.
PASS (bar was 0.05). Deviations split 8 positive / 5 negative — the largest is UBI4
dextrose (+0.042).

The residual +0.008 median offset does not affect the result: **both arms of the Phase-3
comparison go through this same rebuild** (Rosetta's ΔΔG and ours are swapped into
identical frames), so any offset relative to their pkl cancels in the paired difference.

Also fixed a benign `All-NaN slice` RuntimeWarning in `score.py` — positions with no
value for any of the 20 substitutions correctly give NaN, which becomes their -100
sentinel; the warning was noise, not a bug.

### 2026-08-26 — throughput measured: the budget was 2x pessimistic

**prepare 1414 COMPLETED in 21 min** (not the 3–5 h estimated) and built all 25,224
MSAs + queries. Small proteins make short alignments.

**First predict shard: 99 structures in 9 min 52 s** on nodo4 — ~6.0 s/structure
including ~3 min of Boltz startup, so ~4.2 s/structure marginal. The plan's 70–80 GPU-h
came from scaling results/10's 65 s/structure at 398 aa with a 10 s/structure floor;
the real floor is lower.

| | planned | measured |
|---|---|---|
| total | 70–80 GPU-h | **~42 GPU-h** (256 × 9.9 min) |
| wall clock at `%3` | ~1 day | **~14 h** |

Disk on the cluster, projected from the first shard: MSAs 6.4 GB (the bulk), queries
0.2 GB, slim 0.23 MB/structure → **~5.8 GB**, total **~12.4 GB**. `/grupos` has 347 GB
free. `boltz_raw_output` is 16 KB — incremental per-shard slim is deleting raw
correctly (`delete_raw`). 0 failed shards so far.

The local slim store to sync back will be ~5.8 GB against 101 GB free here.

### 2026-08-26 — Phase 0 gate PASSED: their LOPO baselines reproduce

`rf4mave.py` on their own `preprocessed.pkl`, all 39 datasets / 29 proteins:

| model | features | ours | published | Δ |
|---|---|---|---|---|
| null (s̃_exp) | 3 | 0.334 | 0.17 | **+0.164** |
| ΔΔG only (Rosetta) | 1 | **0.249** | 0.25 | −0.001 |
| ΔΔE only (GEMME) | 1 | **0.409** | 0.42 | −0.011 |
| ΔΔG + ΔΔE | 2 | **0.466** | 0.47 | −0.004 |
| position-context | **47** | **0.519** | 0.52 | −0.001 |

**All four baselines pinned by explicit `-f` regexes in their `train.sh` reproduce
within ±0.011** — inside the ±0.02 gate. The harness is trustworthy. The
position-context set came out at exactly **47 features**, matching the paper's stated
count, which independently confirms the decoding (20+1+1 Rosetta, 20+1+1 GEMME, 3 s̃).

The null is the one outlier (+0.164). It is also the only one of the five that their
`train.sh` does **not** define with an explicit feature regex, so what went into their
Figure 2B green box is a guess on our side; the paper quotes it as a *mean*, not a
median. Our 0.334 agrees closely with their own Table S1 "MAVE WT→Mut" column
(median ≈ 0.33), i.e. with the substitution matrix used directly as a predictor. Read
as a definitional difference, not a harness bug — but it is a guess, and is reported
as one. It does not affect the ΔΔG comparison, which is what this experiment is for.

Also worth recording, since the paper leaves these vague and their code settles them:
RF is `n_estimators=150, max_features="sqrt", min_samples_leaf=15`; missing values are
a **−100 sentinel**, not NaN; their `-x 2` drops rows whose own Rosetta *or* GEMME value
is missing, from train and validation alike; and the 47 "position-context" features
decode exactly as 20 + 1 + 1 Rosetta, 20 + 1 + 1 GEMME, and 3 s̃ terms.

`check_frames.py` will verify the other half of the harness — that `score.py`'s rebuild
of those 47 features from the raw PRISM tables (needed so our ΔΔG can be swapped in)
reproduces their feature semantics, by running the same LOPO on both and comparing
per-dataset ρ.

### 2026-08-26 — GPU chain submitted (1414 → 1415 → 1416 → 1417)

`./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3` on cranex:
prepare **1414** (cpu) → predict array **1415** (`0-255%3`, gpu, self-slimming) →
slim sweep **1416** (`afterany`) → features **1417**. Bad nodes excluded up front:
cpu `nodo1,nodo3,nodo5`; gpu `nodo1,nodo3,nodo5,nodo11,nodo12,sauron`. Queue was
empty and GPU nodes idle at submission, so `%3` costs other users nothing.

256 shards ≈ 99 structures each. Expect prepare ~3–5 h (25k mutated MSAs; GLA's 7.5k
took 2 h), then ~70–80 GPU-h of predict.

**Deviation from the plan, stated deliberately:** the plan gated GPU submission on
Phase 0 finishing. Phase 0's random forests are slower than expected (~70 s per fold
for the single-feature models → ~4–6 h total), and Phase 0 validates the *scoring
harness*, which is not needed until Phase 3. The corpus itself was validated
independently through the real code path (`load_dataset` + `prepare_mutations_frame`:
25,213 rows in, **0 dropped**, 11 proteins, exactly 19 mutations per position,
25,224 structures). If Phase 0 turns out to need fixing, that changes `rf4mave.py`,
not the embeddings — so overlapping the two wastes nothing and saves a day.

**Cluster sync:** the cluster checkout is on `main` while this work is on branch
`results/11-12-calibration-and-error-anatomy`. Rather than switch its branch, the two
files the GPU run actually needs were written straight out of the pushed branch:
`git show origin/<branch>:<path> > <path>` for
`experiment_configs/mave_hoie_le200.yaml` and `data/raw/mave_hoie_le200.csv`. This
touches neither the cluster's branch nor its index. The `results/15/` scripts run
locally and are not needed there.

## Log — newest first

### 2026-08-26 — data verified, corpus + leakage map built, Phase 0 running

**Data availability confirmed.** `data.zip` (63 MB) from
`github.com/KULL-Centre/papers/tree/main/2021/ML-variants-Hoie-et-al` (also Zenodo
`10.5281/zenodo.5647207`). `fetch_hoie.py` pulls the 39 merged PRISM tables, their
`preprocessed.pkl` (the built 47-feature tables) and `mut_matrix_alphabetical.npy`
into `data/raw/mave_hoie/` (gitignored, 252 MB).

Checked all 39 datasets: **100 % of variants satisfy `sequence[pos-1] == wt_aa`**
against the header sequence, so they pass `ddg.datasets.prepare`'s validation
untouched. Only 3 of the 39 are true VAMP-seq abundance assays (PTEN 003, TPMT 014,
NUDT15 005); `012_P53_abundance_reversed` is misleadingly named — its header says
growth/phenotype (Giacomelli 2018).

Recomputed the direct per-dataset Spearman baselines from their columns; they
reproduce Table S1. Median |ρ(Rosetta, s_exp)| = **0.301** over all 39.

**Scope decision (user): Tier 1 = the 11 proteins ≤200 aa.** 13 MAVE datasets.
Median |ρ(Rosetta, s_exp)| on this subset is also **0.301**, so the size cap does not
favour or disfavour the stability baseline. Full L×19 saturation (not just measured
variants) because the position-context model needs all 20 substitutions at a
position and it costs only 3.8 % more.

| protein | L | datasets | mutations | structures | scored rows |
|---|---|---|---|---|---|
| CALM1 | 149 | 1 | 2831 | 2832 | 1813 |
| GAL4 | 65 | 1 | 1235 | 1236 | 1196 |
| GmR | 177 | 1 | 3363 | 3364 | 1929 |
| HRas | 189 | 1 | 3591 | 3592 | 3135 |
| IF-1 | 72 | 1 | 1368 | 1369 | 1368 |
| NUDT15 | 164 | 2 | 3116 | 3117 | 5856 |
| PAB1 | 75 | 1 | 1425 | 1426 | 1188 |
| SUMO1 | 101 | 1 | 1919 | 1920 | 1700 |
| UBE2I | 159 | 1 | 3021 | 3022 | 2563 |
| UBI4 | 75 | 2 | 1425 | 1426 | 2575 |
| ccdB | 101 | 1 | 1919 | 1920 | 1176 |
| **total** | | **13** | **25,213** | **25,224** | **24,499** |

`build_corpus.py` → `data/raw/mave_hoie_le200.csv` (corpus) +
`data/raw/mave_hoie_le200_labels.csv` (24,499 rows; 23,444 with Rosetta ΔΔG, 24,445
with GEMME ΔΔE). WT-identity check asserted at build time: **0 mismatches**.

Config: `experiment_configs/mave_hoie_le200.yaml`, derived from `scan_GLA_human.yaml`.
Two deliberate differences: `head.mode: inference` (the label is fitness, not ΔΔG, and
must never occupy the `ddg` column — it is joined back afterwards on
`(uniprot, mutation)`), and **`slim.keep_s: true`** (the scan template ships `false`;
`s` is the only retained field the concat model does not read, but it is what `sdim_*`
features need, and dropping it would mean re-running Boltz. Cost 1.4 → 4.1 GB).

**Leakage map** (`build_homology_map.py`, MMseqs2 15-6f452 at 25 %/30 % id, 80 % cov,
561 pooled sequences = 412 Tsuboyama + 138 FireProt + 11 MAVE):
**UBI4 is the only leaky protein**, at both thresholds, clustering with Tsuboyama's
`1UBQ.pdb`/`1SIF.pdb`/`2MLB.pdb` — i.e. ubiquitin itself. 2,575 of 24,499 scored rows
(10.5 %), 2 of 13 datasets. SUMO1 and UBE2I are **clean** — the ubiquitin *fold*
similarity is below 25 % sequence identity. → report full and UBI4-filtered numbers.
`mmseqs` is on neither this workstation nor cranex; used a static binary via
`MMSEQS_BIN`, same as results/09.

**Phase 0 (harness reproduction), running.** `rf4mave.py` re-implements their LOPO
protocol, decoded from their released code rather than the paper prose:
`RandomForestRegressor(n_estimators=150, max_features="sqrt", min_samples_leaf=15)`;
missing values are the sentinel `-100`, not NaN; their `-x 2` filter drops rows whose
own Rosetta *or* GEMME value is missing, from train and validation alike; features are
selected with the same `str.contains` regexes their `train.sh` passes via `-f`; for
each of the 39 datasets every dataset of the same protein leaves training. Their 47
"position-context" features decode exactly as 20+1+1 Rosetta + 20+1+1 GEMME + 3 s̃.

First result: `null_smave` median ρ = **0.334** vs the paper's quoted 0.17 (which the
text gives as a *mean*, and which is the one baseline `train.sh` does not define via an
explicit `-f` regex). Our 0.334 agrees closely with Table S1's own direct
"MAVE WT→Mut" column (median ≈ 0.33), so the discrepancy is most likely about how the
green box in their Figure 2B was defined, not a harness bug. The four baselines that
*are* pinned by explicit regexes in `train.sh` (ΔΔG-only 0.25, ΔΔE-only 0.42, both
0.47, position-context 0.52) are the real gate — pending.
