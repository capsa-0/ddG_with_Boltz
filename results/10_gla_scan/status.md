# 10_gla_scan — status log

**State:** feature extraction submitted to the cluster. No ΔΔG predictions yet.

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

**Next:**
1. Watch the prepare job — it needs **one** MSA from the ColabFold MMseqs2 server for
   the WT, then builds 7,562 mutated copies (~3.4 GB of a3m; the mutated-MSA loop is
   the slow part of prepare at this scale).
2. Watch the predict array for bad-node startup crashes; predict is resumable, so
   re-running `submit_scan.sh` only redoes leftover work.
3. When `features` finishes → `sbatch slurm/scan_predict.sbatch experiment_configs/scan_GLA_human.yaml`.
4. Then: Boltz-vs-FoldX comparison (Spearman overall and per position), figures,
   `report.pdf`.
