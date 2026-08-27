# 10_gla_scan — status log

**State:** ✅ Done for the subset. The full 398-position scan stays **paused** (too slow); the targeted subset carries the result. Predictions compared against FoldX and against measured residual activity (Lukas et al. 2013); 5 figures, `report.pdf` built.
**Last updated:** 2026-08-27

---

### 2026-08-24 — scan module built, GLA scan generated and submitted

**New code — `ddg/scan/` (the scan module):**
- `mutations.py` — enumerate all L×19 point mutations of a sequence into the
  `minimal` adapter's schema; `AA_ORDER` groups residues by side-chain chemistry
  for the heatmap axis.
- `build.py` — sequence → `data/raw/scan_<name>.csv` + `experiment_configs/scan_<name>.yaml`.
  `--first-residue` records the reference numbering (stored as `scan.first_residue`;
  the CSV stays 1-based because `ddg.datasets.prepare` validates `sequence[pos-1]`).
- `predict.py` — fits regimes A/B/D (concat `wtz`+`mtz`, antisymmetry aug, 5-seed MLP —
  same machinery as results/09) on the labelled corpora and scores the unlabelled scan.
  Emits `scan_predictions.csv`, `scan_matrix_*.csv`, `scan_summary.json`.
- `plots.py` — banded position×residue heatmap, per-position profile, regime-spread panel.
- `__main__.py` — `python -m ddg.scan {build,predict}`.

**Pipeline changes (backward compatible):**
- `ddg/features/build_features.py` now emits configurable feature blocks
  (`feature.blocks`, default `[zdiag, zpool]` — unchanged for every existing config).
  Asking for `[zdiag, zpool, wtz, mtz]` reproduces
  `results/07/build_ablation_features.py` inside the pipeline, so the concat features
  no longer need a separate ad-hoc script.
  **Verified:** rebuilt s669 from its local slim store with all four blocks →
  `max |new − features_ablation.parquet| = 0.0` across all 512 columns; the default
  path still yields exactly the 256 `zdiag`/`zpool` columns.
- `build_features` tolerates unlabelled runs (`head.mode: inference` → `ddg` = NaN).
- `ProjectConfig.feature_blocks` added.

**SLURM:**
- `slurm/submit_scan.sh` — prepare → predict-array (self-slimming) → features.
  **No monolithic slim step** (7.5k structures would be a multi-hour single job that
  strands the chain on failure); bad nodes excluded up front.
- `slurm/scan_predict.sbatch` — CPU job for the scoring step.
- **All `slurm/*.sbatch` switched to the new conda activation** mandated by the cluster
  admins today: `source /home/shared/load-conda` + `conda activate ddG_with_Boltz`,
  replacing `eval "$(conda shell.bash hook)"` + `source activate`. Verified the loader
  exists on the cluster and resolves conda under `$HOME` — which also retires the old
  non-interactive-ssh gotcha (CLAUDE.md updated).

**Scan generated:** `GLA_human`, 398 aa, reported as residues 32–429.
7,562 mutations + 1 WT = **7,563 Boltz structures**. Validated through the real
`minimal` adapter + `prepare_mutations_frame`: **0 rows dropped**, exactly 19 per
position, positions 1–398, WT identity check clean.

**FoldX baseline checked** (`ddg_varmed_by_mutation_foldx.csv`, provided): 7,960 rows =
398 × 20 (19 substitutions + one `X→X` row per position). All 7,562 real substitutions
present exactly once, no duplicates/extras, WT residue matches the sequence at every
position under UniProt numbering. **40 substitutions have no value**: `E58D`, `V137R`,
and all 19 at each of `L428`/`L429`. Range [−5.36, +70.27], mean +3.16 (the high tail
is clash artifact — compare by rank, not RMSE).

**Testing:** end-to-end run of `score_scan` → matrices → all 6 figures on a synthetic
7,562-row GLA scan (random features, real corpora subsampled). Output **joins 1:1 to
the FoldX table on `mutation`** (7,562 both / 0 left / 0 right), positions agree.

**Decisions taken with the user:** all three regimes reported (not just D), numbering
follows the FoldX file (UniProt 32–429, revised from an initial scan-local choice once
that file surfaced), protein id `GLA_human`.

**Submitted (2026-08-24):**

| Job | Step | Detail |
|---|---|---|
| **943** | prepare | RUNNING on nodo10 |
| **944** | predict | array `0-255%2`, `afterok:943`, excl. `nodo[1,3,5,11-12],sauron` |
| **945** | features | `afterok:944_*` |

256 shards (~30 structures each) rather than the usual 128: `/grupos` is at **100 %
(317 GB group-wide headroom)**, and halving the shard size halves peak raw-embedding
disk to ~5 GB (raw z for a 398 aa protein is ~80 MB/structure, deleted per shard by
the incremental slim). Steady-state cost is ~3.4 GB of mutated MSAs + ~2 GB slim store.

**Two submission failures fixed along the way:**
- `git pull` on the cluster was blocked by untracked `fireprot_le200.{csv,yaml}`;
  verified byte-identical (md5) to the committed copies, removed, pulled.
- The predict array was rejected with `Invalid node name specified`: the exclude list
  named **nodo14/nodo15, which no longer exist** (cluster rebuilt — nodes are now
  nodo1–12 + sauron, job IDs reset to 3 digits). Fixed in `predict_array.sbatch`
  (which also now excludes nodo1/nodo5, previously missing), and `submit_scan.sh`
  now filters its exclude list against live `sinfo` so this degrades gracefully.
  Prepare had already been submitted before the failure, so predict/features were
  chained onto job 943 by hand rather than re-running the script.

**Prepare confirmed healthy (~2 h in):** the MMseqs2 server returned the single WT
MSA on the first try (`COMPLETE 150/150`), `mutations.csv` validated on the cluster at
7,562 rows, and the mutant-MSA loop is running at **165 MSA/min → ~43 min** for all
7,562 (~3.7 GB, ~500 KB each). `/grupos` still at 317 GB avail.

**Next:**
1. Watch the prepare job — it needs **one** MSA from the ColabFold MMseqs2 server for
   the WT, then builds 7,562 mutated copies (~3.4 GB of a3m; the mutated-MSA loop is
   the slow part of prepare at this scale).
2. Watch the predict array for bad-node startup crashes; predict is resumable, so
   re-running `submit_scan.sh` only redoes leftover work.
3. When `features` finishes → `sbatch slurm/scan_predict.sbatch experiment_configs/scan_GLA_human.yaml`.
4. Then: Boltz-vs-FoldX comparison (Spearman overall and per position), figures,
   `report.pdf`.


### 2026-08-24 (later) — full scan too slow; pivoted to a targeted subset

**Measured throughput killed the full scan.** Prepare (943) COMPLETED cleanly in
**2 h 00 m** (all 7,563 MSAs + queries, Boltz cache warmed). The predict array then
ran at **~37.6 min per 30-structure shard** (nodo2 ~35 min, nodo6 38–47 min) —
about 65 s/structure plus ~3 min Boltz startup. At the `%2` courtesy throttle that
put the remaining 241 shards at **~75 h (~3 days)**.

Disk, the reason `%2`/256-shards was chosen, turned out to be a non-issue: the whole
run was **5.9 GB**, incremental slim was reclaiming raw correctly (`raw pred: 0`),
and `/grupos` free space actually *rose* to 322 GB. The binding constraint was GPU
concurrency, not disk.

**Decision (user):** rather than take more of the shared GPUs, cut the mutation set
to what fits ~8 h. Full-scan chain 944/945 cancelled; **16 slim shards (~480
structures) preserved** in `data/processed/scan_GLA_human/` — the full scan is
resumable later, since predict skips anything already in the slim store.

**Subset — `scan_GLA_human_hard`, 38 positions / 722 mutations:**
- the **10 positions where the model reportedly overestimates**: 169, 80, 228, 360,
  301, 409, 200, 137, 201, 325 (all in range; WT residues G/V/F/Y/S/N/R/G/G/P)
- **all 31 glycine positions** (the reported weak spot). Note **3 of the 10 flagged
  positions (80, 325, 360) are themselves glycines**, which corroborates the pattern.
- all 19 substitutions at every selected position; 0 rows dropped by `prepare`.

**New module capability** (`ddg/scan/build.py`): `--positions` (comma list + ranges,
in reported numbering), `--wt-residues G`, and `--experiment` to decouple the
experiment name from `--name`. That last one keeps `wt_id = GLA_human` identical
across both scans, so the subset **reuses the full run's base MSA** (no MMseqs2
server call) **and its 16 slim shards** (predict skips whatever is already done).
The config's `scan` block records the selected positions and the header says
PARTIAL, so a subset cannot be mistaken for a full scan.

**Submitted:** prepare **976** → predict **977** (array `0-23%2`) → features **978**.
Estimated ~7.6 h for the array. A monitor is armed on failures / dead dependencies /
shard progress.

**Next:**
1. When 978 finishes → `sbatch slurm/scan_predict.sbatch experiment_configs/scan_GLA_human_hard.yaml`.
2. Compare against FoldX **on the 38 shared positions** (rank correlation — FoldX's
   +70 kcal/mol tail is clash artifact). Check specifically whether Boltz also
   overestimates at the 10 flagged sites and at glycines.
3. Optionally resume the full scan later; 480 structures of it are already banked.

### 2026-08-25 — run stopped early, scan scored, FoldX comparison done

**Stopped at 19/24 shards** (user call — enough data). Before clearing temp dirs,
**salvaged 27 finished structures** from the shard that was mid-flight. Shardless
`slim` (1012) compacted them → `shard_0000.npz`; `features` (1013) then built the
table. Array had slowed to **one** GPU slot (`PENDING (Priority)` — jprieto/jcuellar
ahead of us), so the 8 h estimate was drifting to ~2.5 h more.

**Coverage: 604 / 722 mutations (83.6 %)** over all 38 positions (14–19 substitutions
each). The 118 missing are the queries in the 5 unfinished shards.

**ΔΔG (regime means over the 604):**
| Regime | mean | range |
|---|---|---|
| A — Tsuboyama | +0.28 | [−0.58, +1.65] |
| B — FireProt | +1.24 | [−1.31, +6.65] |
| D — fine-tuned | +0.91 | [−0.52, +4.16] |

Regime agreement (Pearson): A–B 0.634, A–D 0.769, B–D 0.796; mean across-regime SD
0.52 kcal/mol.

**vs FoldX (n = 603 joined; `compare_foldx.py`):**
- **Overall Spearman ρ = +0.504** (Pearson raw +0.514, clipped ±10 +0.523).
- Dropping FoldX's clash tail (< 10 kcal/mol, n = 474) → **ρ = +0.379**: part of the
  headline correlation comes from correctly ranking the extreme clashes as bad.
- **Magnitude is compressed, in the opposite direction to "overestimates":** Boltz
  spans [−0.71, +3.99], FoldX [−2.65, +68.73]. Per-position bias is −1.1 to −25.5
  kcal/mol (Boltz lower). **This is not evidence against the user's observation** —
  that was made against structural intuition, not FoldX, and FoldX is itself
  clash-inflated here (**9/38 positions have mean FoldX > 10 kcal/mol**).

**Neither hypothesis is resolvable at this n:**
- *Flagged positions worse?* median ρ **+0.375** (10 sites) vs **+0.262** (28 others) —
  the flagged sites look **better**, but Mann-Whitney **p = 0.476**.
- *Glycines worse?* per-mutation ρ is higher at G sites (+0.532 vs +0.445), but
  per-position median ρ is lower (+0.246 vs +0.467, **p = 0.077**, only 7 non-G sites).
  The two framings disagree → not a real signal at this sample size.
- **Per-position ρ is dominated by noise:** bootstrap 95 % CI for a single position
  (G150, n = 19) is **[+0.09, +0.83]**. Spread across positions (−0.57 … +0.92,
  SD 0.29) is consistent with sampling noise alone. Per-position rankings here should
  not be interpreted.

**Artifacts:** `scan_predictions_mean.csv`, `scan_summary.json`,
`compare_foldx_{merged,per_position,summary}_mean.csv`,
`figures/01_boltz_vs_foldx_mean.png`, `figures/01_heatmap_mean.png`.

**Next (if pursued):**
1. To test the overestimation claim properly, compare against **measured** ΔΔG or
   structural annotations — not FoldX, which is unusable as ground truth at these sites.
2. Per-position conclusions need the full 19 substitutions **and** more positions;
   at n≈15 nothing is separable.
3. 5 shards' worth of structures remain unpredicted; the full 398-position scan is
   still resumable (`scan_GLA_human`, ~480 structures banked).

### 2026-08-25 — searched for measured ΔΔG ground truth: none exists for GLA

**Leakage check (good news):** GLA / P06280 / the exact WT sequence appears in **none**
of the local corpora (`fireprot_*`, `tsuboyama_*`, `s669`, `ssym`). So regimes A/B/D
have never seen this protein — the scan is a genuine blind prediction. It also means
no free ground truth from the datasets already on disk.

**External search — no ΔΔG dataset exists for α-galactosidase A:**
- **ProThermDB / ThermoMutDB / FireProtDB:** no α-Gal A entries surfaced. Consistent
  with the local check, since the project's FireProt corpus derives from that lineage.
- **The one directly relevant biophysical study** — Andreotti et al., *A thermodynamic
  assay to test pharmacological chaperones for Fabry disease* (PMC3909460) —
  **explicitly declines to report ΔG/ΔΔG**, using urea **C₀.₅** (half-unfolding
  concentration) instead, because ΔG would require m-values they did not measure.
  Only ~4 mutants: **L300F, D244H, Q280K, R301P** (WT C₀.₅ = 1.8 ± 0.1 M urea, pH 7.4).
- **No deep mutational scan / VAMP-seq for GLA** in MaveDB.

**What does exist, all proxies rather than ΔΔG:**
| Source | Measure | Overlap with our 38 positions |
|---|---|---|
| Andreotti et al. | urea C₀.₅ | **R301P** (pos 301) |
| ER-stress/UPR study (PMC9636577) | ER retention, 7 variants | **G360D**, **R301Q** (pos 360, 301) |
| Fabry variant literature | residual activity %WT (class I–IV) | many, but activity ≠ stability |
| Sci Rep 2023 directed evolution | engineered thermostable variants | not systematic single-point |

So the measurable overlap is **~3 mutations at 2 of the 38 positions** — far too few to
validate anything, and none of it on a ΔΔG scale.

**Conclusion:** FoldX remains the only comparator available for this protein, with the
clash-inflation caveat already recorded. Testing the "overestimates at these positions"
claim on a ΔΔG scale is **not possible with public data for GLA**. Options, in order of
strength: (a) validate the claim on a protein that *does* have measured ΔΔG (S669/Ssym
already done — results/09); (b) use ClinVar pathogenic-vs-benign GLA missense variants
as an ordinal proxy (tests pathogenicity, not stability, and is confounded by catalytic
residues); (c) treat residual-activity % as a weak proxy with the same confound.

### 2026-08-25 — FoldX's known failure modes, tested against this structure

**Literature.** FoldX's documented weaknesses land squarely on this position set:
it uses a **rigid backbone**, so "a mutation that introduces a large Van der Waals
clash in a hydrophobic core will either unfold the protein or result in large
backbone conformational changes, but since FoldX does not incorporate backbone
moves, those mutants cannot be predicted" — the clash is instead reported as a huge
positive ΔΔG. Mutations **to Ala/Gly** and **from Pro** are named as the largest
discrepancy classes; proline alone adds ~0.76 kcal/mol of uncertainty
([Bioinformatics 2025](https://academic.oup.com/bioinformatics/article/41/2/btaf064/8003679),
[BMC Bioinformatics 2023](https://pmc.ncbi.nlm.nih.gov/articles/PMC10642056/)).
**31 of our 38 positions are glycines**, so this comparison is close to worst-case
for FoldX.

**Tested on 1R46** (chain A, residues 32–421; identity matches our sequence 390/390):
- **Hypothesis that failed:** that positive-φ (left-handed, Gly-only) sites drive the
  FoldX blow-ups. **The opposite is true** — positive-φ glycines have *median* FoldX
  3.62 kcal/mol vs **15.14** for negative-φ ones. Recorded because it is a real
  negative result, not a slip.
- **What actually drives it: burial.** FoldX per-position mean vs neighbour count
  **ρ = +0.523** (vs −0.227 for φ). Every extreme site is a buried glycine —
  G328 (25 neighbours, +28.8), G147 (25, +26.4), G43 (24, +24.9), G138 (23, +22.7),
  G373 (22, +21.9). Exactly the documented mechanism: buried Gly → any larger side
  chain clashes → rigid backbone cannot relax → runaway ΔΔG.
- **Boltz does not blow up at the same sites** (median +1.05 buried vs +0.55 exposed),
  so the disagreement is FoldX's dynamic range, not a Boltz artifact.

**Filtering to where FoldX should be trustworthy does not rescue agreement:**
| Subset | ρ | n |
|---|---|---|
| all | +0.504 | 603 |
| FoldX < 10 kcal/mol | +0.379 | 474 |
| FoldX ≥ 10 (clash regime) | +0.171 | 129 |
| FoldX < 10 **and** exposed (<22 neighbours) | +0.312 | 324 |

Agreement *falls* as the clash cases are removed — i.e. much of the headline ρ = 0.50
was both methods merely agreeing that buried-glycine substitutions are bad. On the
clean subset the scale mismatch is stark: **regression slope Boltz~FoldX = 0.08**
(Boltz [−0.71, +2.44] vs FoldX [−1.95, +9.66]).

**Bottom line:** FoldX cannot arbitrate the overestimation question on this protein.
Its reliable regime is where the two methods agree *least*, and its unreliable regime
is where the apparent correlation comes from. Ground truth has to come from measured
ΔΔG on other proteins (results/09 corpora), not from FoldX here.

### 2026-08-25 — terminology correction on the discrepancy map

The discrepancy metric was described as one method "ranking worse" than the other.
**Corrected to "ranks more destabilizing"** in `map_discrepancy.py`, its printed output
and the figure title. `delta = pct(Boltz) − pct(FoldX)` measures **disagreement in
relative ordering between two predictors**; with no measured ΔΔG for GLA it cannot say
which is correct, and "worse" invited reading it as an accuracy claim.

Two limits of the metric, now documented in the script:
- Percentiles are computed **within this scanned set**, of which 31/38 positions are
  glycines — not a neutral reference population.
- Rank-normalising both methods **removes disagreement about distribution by
  construction**, concentrating everything into ordering. It therefore cannot see the
  scale mismatch (regression slope Boltz~FoldX = 0.08), which is arguably the largest
  real difference between them.

The closest thing to arbitration available is the **asymmetric mechanistic prior**:
where FoldX is in its clash regime (buried Gly + rigid backbone, established from the
literature and confirmed on 1R46) there is independent reason to distrust it; where
Boltz is the harsher one at exposed sites, there is no such argument and the
disagreement is genuinely unresolved.

### 2026-08-25 — same map on the real kcal/mol scale (figure 03)

Rebuilt the discrepancy map with **raw ΔΔG differences instead of percentile ranks**
(`figures/03_discrepancy_map_raw.png`). Result, quantified:

- **corr(Boltz − FoldX, FoldX) = −0.975.** The raw difference is essentially **−FoldX**;
  it carries almost no independent information.
- Per-position SD: **Boltz 0.74 vs FoldX 7.68 kcal/mol** — FoldX varies **10.3×** more.
- Raw difference range **[−25.51, +0.15]**: on this scale the difference is negative
  almost everywhere, and only one position is even marginally positive.

So the raw-scale plot answers a different question than the rank version. It shows
**how far apart the two scales are** (large, and driven entirely by FoldX's clash
regime at buried glycines), but it cannot show **where they disagree about which
mutations are relatively worse** — Boltz's entire per-position range is thinner than a
single FoldX bar, so every bar just traces FoldX.

Both figures are kept: **03 = scale mismatch** (the physically meaningful comparison),
**02 = ordering disagreement** (the only comparison that is scale-free). Neither is an
accuracy measure — there is still no measured ΔΔG for this protein.

### 2026-08-25 — heatmap corrected (missing data was indistinguishable from neutral)

Two defects fixed in `ddg/scan/plots.py::plot_heatmap`, both of which made the headline
figure misleading:
1. **Missing cells rendered white**, identical to a predicted ΔΔG of ~0 on a diverging
   map — so "not computed" (the 118-mutation coverage gap) read as "predicted neutral".
   Now drawn **grey** via `cmap.set_bad`, with the wild-type cell outlined separately.
2. **The x-axis implied a contiguous stretch of sequence.** A subset scan's columns are
   adjacent on screen but not in sequence; each column is now labelled with its own
   residue (`G328`, `V137`, …) and the axis says so explicitly.

Both apply to any subset scan, not just this one.

### 2026-08-25 — full 398-position scan resumed (jobs 1030/1031/1032)

Resumed `scan_GLA_human` (all 398 positions / 7,562 mutations) at the user's request.

**Banked first, so nothing is recomputed.** The subset run shares `wt_id = GLA_human`,
so its structure keys are identical to the full scan's: copied all 36 of its slim
shards into the full scan store as `hard_*.npz` (non-colliding names; the merge fix in
`ddg/storage/slim.py` now also protects against collisions, but distinct names keep the
provenance readable). Verified by job:

| | |
|---|---|
| queries total | 7,563 |
| in slim store | 1,050 |
| raw not yet slimmed | 13 |
| **already done** | **1,063 (14.1 %)** |
| **remaining** | **6,500** |
| wild-type slice present | yes |

`prepare` re-runs but is a near no-op — all 7,563 MSAs and queries already exist on
disk from the first attempt.

**Submitted:** prepare **1030** → predict **1031** (array `0-255%2`) → features **1032**.

**Honest ETA.** At the measured ~65 s/structure plus ~3 min startup per shard:
~130 GPU-hours. At the polite `%2` throttle that is **~65 h (2.7 days)** with two slots,
and **~130 h (5.4 days)** if cluster contention drops us to one, as happened during the
subset run. Raising `ArrayTaskThrottle` is the only lever that materially shortens this
(`scontrol update JobId=1031 ArrayTaskThrottle=N`) — left at 2 pending a decision.

Disk is not a constraint: the whole experiment is 5.5 GB and `/grupos` has 277 GB free;
incremental per-shard slim keeps peak raw at ~one shard.

### 2026-08-25 — shard 1031_25 failed on a SLURM credential error (data safe)

**What happened.** `1031_25` (nodo4) exited 1. The Boltz prediction **succeeded** — 27
structures written and merged into the canonical predictions dir. The failure was the
*second* `srun` in `predict_array.sbatch`, the incremental slim step:

```
srun: error: Unable to create step for job 1166: Error generating job credential
```

A SLURM infrastructure error (job-step credential), not code, data, or a bad GPU.
**One-off**: 1 of 26 finished tasks, 1 log affected, no recurrence.

**Consequence.** The 27 predictions stayed raw (unslimmed) — harmless, disk is fine
(7.1 GB used, 351 GB free) — but the FAILED task made `1032`'s `afterok:1031_*`
permanently unsatisfiable. `scontrol requeue 1031_25` was **refused** ("Invalid job id")
because the task had already left the queue.

**Fix.** Cancelled 1032 and resubmitted a chain that tolerates the failure:
- **slim 1399** with `--dependency=afterany:1031` (not `afterok`) — runs once the array
  finishes regardless of the one failure. Shardless, so it sweeps up every leftover raw
  folder including that shard's 27.
- **features 1400** with `afterok:1399`.

**Lesson for `submit_scan.sh`:** chaining the post-array steps with `afterok` on the
array makes the whole run hostage to any single transient task failure, even when that
task's *output* is complete. `afterany` + a shardless slim is the resilient pattern,
since slim/features are both idempotent and skip-aware.

**Progress at this point:** 25 COMPLETED (16 of them instant no-ops over pre-banked
structures), 1 FAILED, 2 RUNNING, 228 PENDING. Real shards averaging ~29 min; two GPU
slots active again. Remaining ~233 real shards → **~56 h** at two slots.

### 2026-08-25 — CORRECTION: the ground-truth test already exists (results/12)

Throughout this work I repeatedly proposed "test the glycine/overestimation hypothesis
on S669/Ssym" as **future** work. It had already been done — **`results/12_error_anatomy`**,
on held-out Tsuboyama (n = 12,359, 5-fold GroupKFold) and S669. Correcting the record:

**1. The overestimation observation is CONFIRMED, with ground truth.** results/12 finds
the model's one substantial deficit is the **stabilizing tail**: bias **+0.56** kcal/mol
on held-out Tsuboyama (ρ 0.27, MAE 0.64 against a class spread of only 0.30 — error 2×
the signal), and bias **+1.86** on S669 (ρ 0.11, n = 69). *The model calls stabilizing
mutations destabilizing.* That is exactly "sobreestima", measured against real labels
rather than against FoldX.

**2. It shows up directly in this scan.** Of the 604 scored GLA mutations, only **2**
are predicted below −0.5 kcal/mol (regime D: **1**; regime A: **1**), versus **4.3 %**
of true Tsuboyama labels below that threshold. The scan is effectively incapable of
proposing a stabilizing mutation — the documented blindness, visible in our own output.

**3. Glycines: both results are right, because they measure different things.**
results/12 finds **buried glycines** a genuine weak spot in *magnitude*
(MAE ÷ sd **0.64**, n = 81, vs 0.48 for buried non-Gly). This scan found glycines
*better* in *ranking* (per-mutation ρ +0.532 at G sites vs +0.445). Not a contradiction:
ρ is not degraded at Gly sites in results/12 either. **Magnitude suffers at buried
glycines; ordering does not.**

**4. My burial framing was too generous to the literature.** I attributed FoldX's
blow-ups to burial (correct, ρ = +0.52) and implied burial is hard in general.
results/12 **refutes that for this model**: MAE ÷ sd is 0.49 / 0.48 / 0.48 across
buried / mid / exposed — identical relative accuracy, with the *best* ranking when
buried (ρ 0.79). Burial degrades FoldX, not Boltz.

**5. The cheap fix has been tried and does not work.** `results/13_balanced_loss`:
Balanced MSE removes only **19 %** of the stabilizing bias (0.58 → 0.47) and
significantly *worsens* overall r and MAE; LDS is dominated outright.

**Implication for this experiment.** The right framing for the GLA scan is not "does it
agree with FoldX" but "it is a destabilization ranker with a known stabilizing blind
spot". For Fabry-style work — hunting stabilizing/rescue mutations — that blind spot is
the headline caveat, and it is quantified in results/12, not here.

**Also noted:** results/12 lists `figures/02_*` (the Tsuboyama equivalent panel) as
*pending; regenerating* — an open loose end in that folder, not this one.

### 2026-08-25 — full scan PAUSED at user request (resumable)

Cancelled array **1031** and its downstream **1399**/**1400**. Final tally: **44
COMPLETED** (16 instant no-ops + 28 real), 1 FAILED (`1031_25`, the credential error),
211 CANCELLED.

**Nothing was lost.** Before clearing the temp dirs I salvaged **15** finished
structures from the two shards that were mid-flight — `run_boltz_predictions` wipes
`_predict_shards/*` on restart, so anything left there would have been discarded.

**Banked state:** 80 slim shards + **42 raw** prediction folders (predict skips both on
resume, and each shard's own slim compacts its raw when it next runs). 8.3 GB used,
355 GB free. Local monitor stopped.

**To resume — one command, nothing else needed:**
```bash
./slurm/submit_scan.sh experiment_configs/scan_GLA_human.yaml 256 2
```
It re-runs `prepare` (now a fast skip, since the MSA fix in `14b46c2` stops it
rebuilding the 7,562 mutant alignments), then only predicts what is missing. The chain
now uses `afterany` for the slim sweep, so a single transient shard failure no longer
strands the run.

**Progress at pause:** roughly 2,000 of 7,562 mutations have structures (44 shards done
of 256; ~28 shards' worth of real work beyond the 1,063 pre-banked). Remaining ≈ 212
shards × ~33 min ÷ 2 slots ≈ **58 h**.

### 2026-08-27 — stopped at 29.6 % and finished; the subset's glycine result REVERSES

Resumed (1818–1821, throttle raised to %3), then stopped at the user's request.
**`prepare` took 19m54s vs 1h48m before** — the MSA fix (`14b46c2`) logged
`MSA mutants: 0 written, 7562 already present (skipped)` in 1.3 s, confirming it works.

Salvaged **42** structures from the three in-flight shards. The shardless slim then hit
one **corrupt** prediction (`GLA_human_R7W`, truncated when its shard was killed
mid-write), deleted it and aborted — the corruption guard behaving correctly. Re-ran:
slim OK (raw 9.8 GB → 0, 94 shards), features OK.

**Final dataset: 2,239 mutations over 177 positions (29.6 % of 7,562), 3.7× the subset
and no longer glycine-dominated — glycines are 505/2,238 (23 %) vs 82 % before.**

**Predictions:** A +1.28 [−1.21, +4.64], B +1.17 [−3.95, +6.40], D +1.33 [−1.05, +5.52].
Regime agreement rose sharply (Pearson 0.78–0.92 vs 0.63–0.80) and across-regime SD fell
to **0.29** (from 0.52) — the wider corpus makes the three regimes converge.

## The subset's headline finding was a small-sample artifact

| | subset (604, 82 % Gly) | **full (2,238, 23 % Gly)** |
|---|---|---|
| overall ρ | +0.504 | **+0.595** |
| glycine ρ | +0.532 | **+0.458** |
| non-glycine ρ | +0.445 (n=110) | **+0.639** (n=1,733) |
| flagged median ρ | +0.375 | **+0.261** |
| other positions median ρ | +0.262 | **+0.404** |

**Both comparisons flip.** With only 110 non-glycine mutations the subset made glycines
look *better* than the rest; with 1,733 they are clearly **worse** (+0.458 vs +0.639).
That now **agrees with results/12**, which found buried glycines a genuine weak spot in
magnitude — and removes the discrepancy I had reconciled by appealing to
ranking-vs-magnitude. The honest reading is simpler: the subset was underpowered.

Likewise the **10 flagged positions now agree with FoldX *worse* than the rest**
(median ρ +0.261 vs +0.404), the opposite of the subset. Consistent with the user's
original observation, though still not significant at position level
(Mann-Whitney **p = 0.120**; glycine-vs-not **p = 0.058**) — per-position ρ stays noisy.
At *mutation* level, where the power is, the glycine gap is unambiguous.

**Other numbers:** removing FoldX's clash tail barely changes ρ now (+0.602 at
FoldX < 10, vs +0.595 overall) — unlike the subset, where it *fell* from 0.504 to 0.379.
That was the glycine bias too. Scale mismatch persists but is milder: FoldX per-position
SD 5.36 vs Boltz 0.82 (6.5×, was 10.3×); raw difference still −0.948 correlated with
FoldX alone. Stabilizing predictions: 173/2,238 (7.7 %), only 20 below −0.5 — the
results/12 stabilizing blindness, still visible.

**Artifacts refreshed:** `scan_predictions_mean.csv`, `scan_summary.json`,
`compare_foldx_*_mean.csv`, `discrepancy_by_position.csv`, `boltz_minus_foldx.pdb`,
all four figures.

**Remaining:** 5,323 mutations unpredicted (221 positions untouched, 177 covered). The
run is resumable exactly as before.

### 2026-08-27 — report.pdf written

Paper-facing write-up built by `build_report.py` (4 pages, A4): motivation, methods,
results with the three figures, interpretation, limitations, conclusion. Every number is
recomputed from the committed tables at build time, so the report cannot drift from the
data.

Per `results/guidelines.md` it contains **no provenance or plumbing** — no job IDs, node
names, paths, shard/resume history, the corrupt-file incident, or the earlier
glycine-biased subset. Verified mechanically: zero hits for those terms in the extracted
text. That history stays here in `status.md`; the paths stay in the README table.

One deliberate framing choice: the report presents the {n} scored substitutions as *the*
corpus and lists partial coverage under Limitations, rather than narrating how the corpus
came to be that size.

### 2026-08-27 — re-checked for public measured ΔΔG on GLA: still none (search widened)

Re-ran the ground-truth search of 2026-08-25 with wider terms (thermal stability / Tm /
DSF, urea unfolding ΔG, deep mutational scans 2025–2026, ProThermDB/ThermoMutDB/MaveDB).
**Conclusion unchanged: no experimental ΔΔG in kcal/mol exists for α-Gal A**, for any
substitution in the FoldX table. What the widened search added:

- **Lukas et al. 2013, PLoS Genet (PMC3731228)** — the largest measured GLA set:
  **158 missense variants** functionally characterised in HEK293H (residual activity
  %WT in 4 classes, DGJ responsiveness, Western-blot degradation). **No ΔΔG, no Tm.**
  At our current 29.6 % scan coverage the expected overlap with the 2,238 scored
  substitutions is ~45–50 mutations — the only measured set large enough to be worth
  joining, as an *ordinal activity* proxy, not a stability label.
- **IJMS 2025 "Bioinformatics-Driven Multi-Factorial Insight into α-Galactosidase
  Mutations" (PMC12193200)** — purely computational (FoldX + AlphaMissense + EVE + MD),
  pulls its FoldX ΔΔG from the **ProtVar API**. Notable because that is the likely
  provenance class of the supplied `ddg_varmed_by_mutation_foldx.csv`; it is *not* an
  independent experimental source.
- Engineered-variant Tm shifts exist (e.g. PubMed 36959353, +2/+4 °C vs WT, +5.5 °C with
  migalastat) but are multi-point designs, not systematic single substitutions.

Spot-check of the literature-measured variants against our tables (`mutation`: FoldX /
Boltz mean, "–" = not yet scored): L300F 4.44/1.39, R301P 5.99/2.21, R301Q 3.76/2.21,
A143T 4.79/0.95, G373S 9.74/1.89, D266N 0.83/0.48; D244H, Q280K, G360D, N215S in FoldX
only. So the ~4 urea-C₀.₅ variants of Andreotti et al. give **2 scored overlaps**
(L300F, R301P) — unchanged from the earlier entry, and still too few to validate.

**Standing recommendation:** the ΔΔG-scale validation stays on S669/Ssym (results/09).
The only new option worth considering for GLA is joining Lukas 2013 residual activity as
an ordinal proxy (rank-based, confounded by catalytic-site variants where loss of
activity ≠ loss of stability).

### 2026-08-27 — activity proxy tested: Lukas 2013 residual activity vs the scan

Followed through on the option left open in the entry above. **Source:** Lukas et al.
2013, *PLoS Genet* 9(8):e1003632, **Supplementary Table S1** — 159 missense GLA variants
expressed in HEK293H with residual α-Gal A activity (% WT, ± the chaperone DGJ). The
table ships as a legacy Word `.doc`; `compare_lukas.py --fetch` downloads it and parses
the OLE piece table directly (no `libreoffice`/`antiword` — both failed on this file),
writing `lukas2013_activity.csv`. Numbering matches ours (UniProt precursor numbering),
so the join is on the raw mutation string.

**Overlap:** 157 variants in the mature chain 32–429; 156 in the FoldX table; **45 also
scored by the scan** (the scan is at 29.6 % coverage).

| Test (n = 45 unless noted) | ρ | p |
|---|---|---|
| Boltz `ddg_mean` vs activity | **−0.305** | 0.042 |
| Boltz, active-site residues dropped (n = 42) | −0.338 | 0.028 |
| FoldX, same 45 | −0.275 | 0.068 |
| FoldX, active-site dropped (n = 42) | −0.344 | 0.026 |
| paired ρ(Boltz) − ρ(FoldX) | −0.030, CI [−0.297, +0.237] | P(Boltz better) = 0.59 |
| dead (0 % activity, n = 27) vs alive, Boltz median | +1.72 vs +1.09 | MWU 0.049 |
| dead vs alive, FoldX median | +3.01 vs +2.43 | MWU 0.112 |

Boltz ρ 95 % CI = [−0.561, −0.011] — excludes zero, but only just. Per regime:
A −0.334, D −0.315, B −0.213.

**Three things this does and does not say:**
1. **Right sign, real but weak signal.** Higher predicted ΔΔG does track lower measured
   activity, and it survives dropping the 3 active-site variants (it strengthens, as
   expected if the catalytic confound is what activity adds on top of stability).
2. **Boltz ≈ FoldX here.** On the mutations both cover, the difference is noise. The
   headline ρ = −0.491 for FoldX over all 157 is **not** a FoldX win: FoldX scores
   ρ = −0.597 on the 111 variants we have *not* scored and only −0.275 on our 45, i.e.
   **our 45 are a harder-than-average subset for both methods** — a consequence of the
   scan's non-random position coverage, not of the predictor. Any comparison must be on
   the shared subset.
3. **It is not a ΔΔG validation.** Activity is zero-inflated (27/45 are exactly 0 %) and
   a catalytic-site variant can be dead while folded. This bounds the claim at "ranks
   loss-of-function better than chance", not "predicts stability accurately".

**Committed:** `compare_lukas.py`, `lukas2013_activity.csv`, `compare_lukas_merged.csv`,
`figures/04_lukas_activity.png`. README headline + provenance table and
`figures/README.md` updated. `report.pdf` **not** regenerated — worth folding this in
when the remaining 5,323 mutations land, since coverage is what limits n here (finishing
the scan would take the overlap from 45 to ~157).

### 2026-08-27 — active-site set corrected (was from memory), figure 04 restricted, §3.5 added to report

**Correction.** The `ACTIVE_SITE` constant in the first version of `compare_lukas.py` was
written from memory, attributed to Garman & Garboczi 2004, and **was wrong** — it contained
172 and 297, which are not in the pocket, and was missing 143, which is. Replaced with a set
derived from two verifiable sources and recomputable on demand:

- **UniProt P06280 feature table:** `ACT_SITE 170` (nucleophile), `ACT_SITE 231` (proton
  donor), `BINDING 203..207` (substrate).
- **PDB `1R47`** (α-Gal A with galactose bound): every residue with a heavy atom within
  **5.0 Å** of the catalytic-pocket ligand `GAL A1101` / `B1103`, unioned over both monomers.
  The NAG/MAN/FUC hetero-atoms are N-glycans and are excluded by residue name. 1R47 numbers
  the mature chain 32–429, i.e. UniProt numbering, so no offset is applied anywhere.
  Sanity checks: D170, E203, Y207, D231 come back with the right identities.

Union = {47, 92, 93, 134, 142, 143, 168, 170, 203–207, 227, 231, 266, 267} (17 positions).
`compare_lukas.py --pdb 1R47.pdb` recomputes the shell, prints all three sets and **fails
loudly** if the derivation stops matching the committed constant.

**Effect on the numbers.** 4 of the 45 shared variants are now flagged (`C142R`, `A143P`,
`A143T`, `D170N`), not 3, so the clean subset is **41, not 42**:

| Test | before (wrong set, n=42) | now (n=41) |
|---|---|---|
| Boltz, no active-site | −0.338 (p 0.028) | **−0.328 (p 0.036)** |
| FoldX, no active-site | −0.344 (p 0.026) | **−0.343 (p 0.028)** |
| paired ρ(Boltz) − ρ(FoldX) | not computed on subset | +0.015, CI [−0.250, +0.286] |
| dead vs alive, Boltz | — | +1.76 vs +1.10, MWU p 0.042 |
| dead vs alive, FoldX | — | +3.36 vs +2.40, MWU p 0.061 |

The full-45 numbers are unchanged (−0.305 / −0.275). Conclusion unchanged: indistinguishable.

**Figure 04 rebuilt** showing only the 41 non-active-site variants (they are dropped, not
marked — at those positions activity is not a stability proxy at all). Trend bars are the
**mean** per tercile, not the median: 24 of 41 sit at exactly 0 % activity, so medians are 0
in most bins and hide the signal. Each bar is labelled with its zero count — for Boltz that
runs 9/14 → 4/13 → 11/14, i.e. **not monotone**: its lowest-ΔΔG tercile still holds 9 dead
variants. FoldX is monotone (6/14 → 7/13 → 11/14). Worth stating plainly rather than
smoothing over. No least-squares line on either panel: the reported statistic is a rank
correlation and an OLS slope on FoldX would be set by the clash tail.

**`report.pdf` rebuilt** (still 4 pages) with a new **§3.5 "An external check against
measured data"** + Figure 4, and the caveat box and Limitations amended to say the activity
proxy constrains loss of function, not folding free energy. Per `guidelines.md` the section
cites the study and states the method but carries no provenance — the `.doc` parsing, the
PDB download and this correction stay here. Mechanically re-checked for leaked plumbing
terms: none.

**Next step unchanged:** finishing the remaining 5,323 mutations takes the overlap from 41
to ~150 and is the only thing that would move this from suggestive to conclusive.

### 2026-08-27 — tested the "FoldX ranks glycines higher within its own spread" reading of figure 01

Question raised from the scatter: glycines and flagged positions sit **below-right of a
hypothetical fit** (high FoldX, mid Boltz). Does that quantify as FoldX ranking them higher
*within its own range* than Boltz does?

**Where they actually sit** (`compare_foldx_merged_mean.csv`, n = 2,238):

| group | n | median FoldX | median Boltz | pct FoldX | pct Boltz |
|---|---|---|---|---|---|
| glycine | 505 | 4.31 | 1.26 | 70.4 | 56.1 |
| non-glycine | 1,733 | 1.85 | 1.03 | 41.1 | 47.5 |
| flagged | 160 | 4.30 | 1.25 | 70.3 | 55.7 |

So yes — below-right, and the gap is in the *percentiles*, not only in kcal/mol.

**The metric already existed and was never split this way.** `map_discrepancy.py` computes
`delta = pct(Boltz) − pct(FoldX)` per mutation — each method ranked within its own spread,
which is exactly the "within its own range" question — but only ever aggregated it *per
position* (figure 02). The tests we had run by residue type were on per-position **Spearman**
(agreement in ordering), a different quantity: it cannot see a systematic shift, only scatter.
Running the split on `delta` is new.

**Result — glycine (negative = FoldX ranks it higher than Boltz does):**

| test | median delta | n | p | Cliff's d |
|---|---|---|---|---|
| glycine vs non-glycine | **−13.1 vs +5.4** (gap −18.5 pts, CI [−20.9, −16.7]) | 505 / 1733 | 4.5e−60 | **−0.477** |
| same, FoldX < 10 (re-ranked) | −15.7 vs +4.2 | 379 / 1664 | 7.3e−45 | −0.462 |
| same, FoldX < 5 (re-ranked) | −19.8 vs +3.2 | 288 / 1420 | 2.9e−37 | −0.476 |

**It is not the clash tail** — dropping it leaves the effect intact, slightly stronger. And it
is a *shift*, not extra noise: mean |delta| is 21.9 for glycines vs 19.3 for the rest, i.e.
the dispersion is comparable while the centre moves ~18 points.

**The flagged-position effect is mostly the glycines inside it:**

| test | median delta | n | p | Cliff's d |
|---|---|---|---|---|
| flagged vs rest, **non-glycine only** | +3.6 vs +5.7 | 114 / 1619 | 0.040 | −0.115 |
| flagged vs rest, **glycine only** | −19.7 vs −12.2 | 46 / 459 | 0.005 | −0.252 |

Among non-glycines, being flagged is worth ~2 percentile points (negligible). Among glycines
it does add a real extra shift. So the original ten-position hypothesis is **not independent
of the glycine effect** — it is largely a restatement of it, plus a genuine extra among the
flagged glycines specifically.

**What this does and does not license.** `delta` is zero-sum by construction (both percentile
means are 50), so glycines being negative *forces* non-glycines positive; the statement is
strictly relative — "FoldX ranks glycines higher than Boltz does" — and cannot say which
method is wrong. Two independent facts pull in opposite directions and the metric cannot
separate them: FoldX holds the backbone rigid, which is worst exactly where glycine is
replaced; and results/12 finds **buried glycines are also one of Boltz's genuine weak spots**
on labelled data (MAE ÷ sd 0.64 vs 0.48 for buried non-Gly), i.e. Boltz plausibly
*under*-ranks them at the same time. Both can be true and the observed 18-point gap is
their sum.

Numbers computed ad hoc from the committed table; **not yet folded into
`map_discrepancy.py` or the report**. Worth doing if this becomes a claim in the paper.

### 2026-08-27 — "% que cae debajo de la recta": la recta ajustada no sirve, la diagonal percentil sí

Follow-up: would fitting a line to Boltz~FoldX and reporting the share of each group below
it add anything? **The share is the right intuition; a fitted line is the wrong ruler.**
The same statistic under five ways of drawing the line (% of each group *below* it):

| recta | glicina | flagged | flagged no-Gly | flagged Gly | resto |
|---|---|---|---|---|---|
| **diagonal percentil (sin ajuste)** | **77.8** | 58.8 | 43.0 | **97.8** | 38.2 |
| OLS crudo (pend 0.055) | 58.2 | 51.2 | 49.1 | 56.5 | 56.3 |
| OLS clip ±10 (pend 0.168) | 67.3 | 58.1 | 46.5 | 87.0 | 51.9 |
| OLS symlog (pend 0.590) | 68.5 | 55.6 | 44.7 | 82.6 | 52.6 |
| Theil-Sen (pend 0.169) | 73.3 | 57.5 | 45.6 | 87.0 | 51.6 |

The glycine number moves 58 → 78 and flagged-Gly 57 → 98 on fit choice alone. Worst case is
plain OLS: at slope 0.055 the line is nearly horizontal, so "below the line" degenerates into
"low Boltz" and the contrast against the rest vanishes entirely (58.2 vs 56.3).

**Use the percentile diagonal** (`pct_boltz = pct_foldx`) — no fit, scale-free, and it is
exactly the "within its own range" question, i.e. `sign(delta)` from the entry above:

| grupo | debajo | % | IC95 |
|---|---|---|---|
| glicina | 393/505 | 77.8 | [74.0, 81.2] |
| flagged | 94/160 | 58.8 | [51.0, 66.1] |
| **flagged no-Gly** | 49/114 | **43.0** | [34.3, 52.2] |
| **flagged Gly** | **45/46** | **97.8** | [88.7, 99.6] |
| resto (no-Gly, no-flag) | 618/1619 | 38.2 | [35.8, 40.6] |

Global base rate 47.4 % (not 50 — ties). Same conclusion as the `delta` test, now in a form
that reads off the scatter: **45 of the 46 flagged glycines sit below-right**, while flagged
non-glycines (43.0 %) are within noise of the rest (38.2 %).

Per-group agreement, for completeness (different axis — scatter, not shift): Spearman
ρ = +0.595 overall, +0.458 at glycines, +0.639 at non-glycines.

Caveat: the percentage binarises `delta` and throws away magnitude; Cliff's d = −0.477 is the
continuous version and is the number to quote statistically. The % is for the narrative.

### 2026-08-27 — percentile diagonal added to figure 01; §3.4 of the report re-derived

Implemented the diagonal + group shares in `compare_foldx.py` (not a throwaway script):

- `percentile_shift(scored)` — share of each group below `pct(Boltz) = pct(FoldX)`, with
  Wilson CIs and median delta. Its docstring records **why there is no fitted line**: the
  OLS slope of Boltz on FoldX is ~0.055, so "below the line" collapses into "low Boltz"
  and the contrast dies (58 % vs 56 % instead of 78 % vs 38 %).
- `scored` now carries `pct_boltz` / `pct_foldx` / `delta`, and those columns are written
  into `compare_foldx_merged_mean.csv` (extra columns; `build_report.py` reads by name and
  is unaffected). New table `percentile_shift_mean.csv`.
- **Figure 01 relaid out** from 1×2 to a gridspec: the two scatters share the top row, the
  per-position trace spans the full bottom (it has one tick per scanned position and was
  cramped at half width). New top-right panel = percentile space, dashed diagonal, both
  triangles labelled ("above: Boltz ranks it higher" / "below: FoldX ranks it higher"), and
  a monospace box with the five group shares.

Regenerated with `compare_foldx.py --scan scan_predictions_mean.csv --regime mean`. Shares
match the ad-hoc computation exactly (glycine 77.8, flagged-Gly 45/46 = 97.8, flagged
non-Gly 43.0, rest 38.2, all 47.4).

**`report.pdf` (now 5 pages).** Figure 2's caption had to be rewritten anyway — it described
a two-panel figure that no longer exists. Also extended §3.4: it presented the flagged
positions and the glycines as two separate findings, which the panel now visibly contradicts
on the same page. The new paragraph states that the flagged-position hypothesis is largely a
restatement of the glycine effect, with a real extra shift only where they coincide. All
figures are numbers pulled from `percentile_shift_mean.csv` at build time, so the prose
cannot drift from the panel.

Still open, unchanged: this is a *relative* statement between two predictors and cannot say
which is wrong; results/12's buried-glycine weakness for Boltz and FoldX's rigid backbone
both push the same direction and the metric cannot separate them.
