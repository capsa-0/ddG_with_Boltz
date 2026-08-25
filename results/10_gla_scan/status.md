# 10_gla_scan — status log

**State:** full scan **paused** (too slow); a targeted 722-mutation subset (`scan_GLA_human_hard`) is running instead — jobs 976 → 977 → 978. No ΔΔG predictions yet.

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
