# Experiment log

Living record of what we ran, how, why, and what happened — the source of truth
for explaining this work to colleagues. Methodology sits at the top; the
chronological run log and planned experiments follow. Pairs with
`docs/benchmark_plan.md` (the holdout methodology in detail).

---

## 1. Goal

Predict single-point mutation ΔΔG from Boltz-2 internal embeddings and, crucially,
measure **how well the prediction generalizes** — to unseen proteins, unseen
sequence clusters, de-novo designs, and unseen chemistries. The generalization
story (how much accuracy drops off the random-CV baseline under each holdout) is
the scientific result, not the single headline correlation.

## 2. Data / corpus

Source: Tsuboyama 2023 mega-scale single mutants (`data/raw/tsuboyama_single_mutants_ddg.csv`)
— 389,068 mutations, 412 proteins (256 natural PDB + 156 designed/de-novo),
lengths 32–72 aa, all 380 substitutions covered.

Corpus design (`ddg_datasets/build_benchmark_corpus.py`): **keep all 412 proteins,
subsample K mutations/protein** ("wide but shallow"). Rationale: the scarce
resource for the top holdouts (protein / cluster / de-novo) is *number of
proteins*, not depth per protein; the chemistry/substitution holdouts pool across
proteins and still see 360+/380 substitutions even when shallow.

| Corpus | K | mutants | Boltz preds (+WT) | file |
|--------|---|---------|-------------------|------|
| fast | 30 | 12,360 | ~12,772 | `data/raw/tsuboyama_bench_fast.csv` |
| wide | 90 | 37,080 | ~37,492 | `data/raw/tsuboyama_bench_wide.csv` |

The CSVs carry helper columns `is_natural` and `chem_category` (the benchmark
recomputes all split labels from `wt_id`+`mutation` anyway, so they are just
convenience).

## 3. Pipeline

`python -m ddg run <config> --step prepare|predict|slim|features`, orchestrated on
SLURM by `slurm/submit_all.sh` (prepare→predict array→slim→features, chained with
`afterok`). Steps:

1. **prepare** — dataset → MSAs (MMseqs2) → mutated/flattened MSAs → Boltz YAML.
2. **predict** — Boltz `--embeddings_only`; keeps trunk outputs `s`, `z`,
   `pdistogram` for WT and mutant. Sharded across GPUs, resumable.
3. **slim** — compact per-mutation slices: full `s` (L×Ds), plus only the
   mutation-position *row* of `z`/`pdistogram` (all the extractor ever reads),
   float16. `delete_raw: true` removes raw NPZs as it goes.
4. **features** — WT-vs-mutant feature engineering → `features_summary.parquet`.

### What the features actually are (s vs z)

The engineered feature vector is dominated by **`s`**: `local_s_dim_{d}_signed_diff`
is the full per-dimension WT−mut difference of the single track at the mutated
residue (the only raw per-dim signal kept), plus summary stats of local/neighbor
`s`. **`z` and `pdistogram` contribute only summary statistics** of the mutation
row/diagonal (mean/std/gini/entropy + KL), never raw values. UMAP in the old
`exploration` plots runs on this engineered vector, *not* on raw `z`. → motivates
the **s-ablation** (§6).

## 4. Modeling & holdouts

`ddg.evaluation` trains a regressor on the feature table and evaluates ΔΔG
prediction under each holdout (train on split-train, predict split-test):
random 10-fold (baseline), protein (5-fold group CV), cluster (homology, needs a
map), de-novo (natural↔designed), leave-one-substitution-out, source/target
residue, chemistry categories. Metrics: Pearson/Spearman/RMSE/MAE, pooled + per-unit
mean±SD. See `docs/benchmark_plan.md`.

**Current limitation (to fix before wide):** single model (SVR), fixed
hyperparameters, no tuning/nested-CV, no model comparison. See §6.

---

## 5. Run log

### 2026-07-15 — fast corpus (`tsuboyama_bench_fast`)
- **prepare** (job 211885): OK. 412 MSAs, 12,772 queries. MSA fetch ~15 min (9
  batches of 50); base MSA is per unique WT.
- **predict** (array 211886_0..5, 6 shards %3): OK. All shards 100%, ~0.8 it/s,
  ~45 min each. 12,772 embeddings.
- **slim** (job 211887): **FAILED** — `zipfile.BadZipFile: Bad CRC-32 for z.npy`.
  → **root cause: transient NFS read glitch, not real corruption** (see recovery).
- **features** (job 211888): never ran — stuck `DependencyNeverSatisfied` (zombie,
  later cancelled).

#### Recovery (2026-07-15, evening)
- Hardened `slim` (commit 5e87649): catches `BadZipFile`/`EOFError`/`zlib` per
  structure, deletes the corrupt folder so predict-resume regenerates it, and
  reports all bad structures in one pass instead of crashing on the first.
- Re-ran **slim** (job 211901): **COMPLETED**, 12,771 structures, 867 MB shard,
  all raw deleted. **Found zero corrupt files** → confirms 211887 was a transient
  read glitch. No predict regen needed.
- **features** (job 211902) + **eval** (job 211903, `afterok`): submitted; features
  ~1.9 it/s over 12,360 samples (~1.5–2 h). [results pending]

### Known issues / observations
- **float16 overflow in `s`**: slim casts `s` (and `zrow`) to float16; some values
  exceed 65504 → `inf` in the slim store → those features get imputed downstream.
  Pre-existing. If the s-ablation (§6) drops `s`, this disappears; otherwise
  consider float32 for `s`.
- **12,771 vs 12,772**: one structure key deduped (no "missing embeddings"
  warning) — negligible, at most one mutation dropped.
- **conda in non-interactive SLURM submits**: `cpu_step.sbatch` uses
  `conda shell.bash hook`, which needs `conda` on PATH. Interactive shells have it;
  a non-interactive `ssh ... sbatch` must `source .../conda.sh` first, else the job
  dies at activation (hit once, job 211900).

---

## 6. Planned experiments / modifications

1. **Analyze fast results** — ceiling (random CV) vs generalization drop
   (protein/cluster/de-novo), per-unit distributions, hard substitutions/chemistries.
2. **s-ablation** — does the `s` embedding actually help? Compare holdout metrics
   *with* vs *without* the s-derived feature groups, and check s-feature
   correlations/importance. If it adds nothing, **drop `s`** (simpler features +
   removes the float16-overflow issue + smaller slim store). Method: a feature-group
   toggle in `ddg.evaluation`.
3. **Rigorous modeling for wide** — hyperparameter tuning (nested CV) and a model
   bake-off (SVR / MLP / gradient boosting / Ridge baseline) so the wide corpus
   gets a defensible, tuned result for the tutor.
4. **wide corpus run** (`tsuboyama_bench_wide`, ~37.5k preds, ~5–6 h) once the
   above are settled.
