# ΔΔG generalization benchmark plan

How we turn **one** Boltz feature-extraction run into a full suite of
generalization experiments, and how we report them.

## Core idea

Boltz embeddings are extracted **once per mutation** and stored in
`data/processed/<experiment>/features_summary.parquet` (one row per mutation,
WT-vs-mutant difference features + `ddg` target). **Every holdout below is a
different train/test split of that same table**, evaluated by re-fitting the
downstream regressor (SVR / MLP). None of them require re-running Boltz.

So the only upfront cost is choosing the corpus. We use the Tsuboyama single
mutants because it is dense (near-complete DMS) and spans natural *and* de-novo
proteins.

## The corpus

Built by `ddg_datasets/build_benchmark_corpus.py`: **all 412 proteins**,
subsampled to K mutations/protein (stratified by substitution).

| Corpus | K | mutants | Boltz preds (+WT) | ~wall time (3 GPU) | file |
|--------|---|---------|-------------------|--------------------|------|
| fast | 30 | 12,360 | ~12,772 | ~1.5–2 h | `data/raw/tsuboyama_bench_fast.csv` |
| wide | 90 | 37,080 | ~37,492 | ~5–6 h | `data/raw/tsuboyama_bench_wide.csv` |

Rationale: the scarce resource for the highest-value holdouts (protein / cluster
/ de-novo) is **number of proteins**, not mutations per protein — so we keep all
412 and go shallow. Substitution/chemistry holdouts pool across proteins, so
they still see 360+/380 substitutions even in the fast corpus.

The corpus CSV carries two helper columns beyond the four the `dms` adapter
reads (`protein_id, wt_sequence, mutation, ddg`):
- `is_natural` — 1 if `protein_id` matches a PDB code (256 proteins), else 0
  (156 designed/de-novo). Drives the de-novo holdout.
- `chem_category` — precomputed substitution class (drives chemistry holdouts).

## Holdouts we will actually run tomorrow

Ordered by scientific value. All are splits of `features_summary.parquet`.
"Unit" = what is held out; "How" = split construction.

| # | Holdout | Unit | How | Metric summary | Notes |
|---|---------|------|-----|----------------|-------|
| 0 | Random baseline | mutation | 10-fold random CV | mean±SD | interpolation ceiling |
| 1 | **Protein holdout** | `protein_id` | 5-fold CV over proteins (a protein is never in train+test) | mean±SD **+ per-protein distribution** | generalization to new proteins |
| 2 | **Cluster holdout** | seq cluster | MMseqs2 cluster WT seqs @30% id, 5-fold CV over clusters | mean±SD + per-cluster | out-of-homology (the convincing one) |
| 3 | **De-novo holdout** | design/natural | train on `is_natural==1`, test on `is_natural==0` (and reverse) | single split | out-of-evolution transfer |
| 4 | Leave-one-substitution-out | A→B | for each of ~360 substitutions, hold it out | mean±SD across substitutions | transferable residue chemistry |
| 5 | Leave-source-residue-out | source AA | 20 folds, hold out all X→\* | per-source-AA table | understanding of original residue |
| 6 | Leave-target-residue-out | target AA | 20 folds, hold out all \*→X | per-target-AA table | understanding of introduced residue |
| 7 | Chemistry category holdouts | `chem_category` | hold out one class at a time: X→P, P→X, X→G, G→X, hydrophobic↔polar, charge inversions, neutral↔charged | one number each | physicochemical rule transfer |

For #1–#3 we report **not just the mean** but the distribution across units
(per-protein / per-cluster Pearson), because a global 0.75 can hide a few
families near 0. That distribution table is the most informative thing for the
tutor.

## Metrics (report all, for every holdout)

Pearson r, Spearman ρ, RMSE, MAE — each as **mean ± SD over folds** — plus
**N mutations**, **N proteins**, and **N clusters/units** where applicable.

## Not feasible by tomorrow (future work — state explicitly)

These need annotations we do not have wired up yet; list them as planned, don't
fake them:
- **Fold / family holdout** — needs SCOP/CATH or Pfam annotation per WT.
- **Buried/surface, helix/sheet/loop** — needs per-residue DSSP/SASA from the
  reference structures (embeddings-only mode skips the structure head, but the
  original Tsuboyama PDBs could supply this later).
- **Compound holdouts** (protein+substitution, fold+chemistry, de-novo+chemistry)
  — cheap to add once the single-axis splits above are in place.
- **C→X holdout** — data-limited: cysteines are rare, so the corpus has only a
  handful of C→X points. Note the low N rather than reporting a noisy number.

## How to run it

```bash
# 1. (already done) build the corpus CSVs
python ddg_datasets/build_benchmark_corpus.py --k 30 --out data/raw/tsuboyama_bench_fast.csv
python ddg_datasets/build_benchmark_corpus.py --k 90 --out data/raw/tsuboyama_bench_wide.csv

# 2. extract embeddings on the cluster (prepare -> predict array -> slim -> features)
#    N shards, at most 3 GPUs at a time:
./slurm/submit_all.sh experiment_configs/tsuboyama_bench_fast.yaml 6 3
# or the wide corpus overnight:
./slurm/submit_all.sh experiment_configs/tsuboyama_bench_wide.yaml 12 3

# 3. build features (once predict finishes)
python -m src.exploration.explore_features --config experiment_configs/tsuboyama_bench_fast.yaml

# 4. run the holdout splits downstream (notebook / eval script) over
#    features_summary.parquet using protein_id, is_natural, chem_category,
#    and the MMseqs2 clusters.
```

### Cluster labels for holdout #2

MMseqs2 clustering of the 412 WT sequences at 30% identity (reuses the existing
ColabFold client in `external/mmseqs.py`) produces a `protein_id -> cluster_id`
map. Assign whole clusters to CV folds so no cluster is split across train/test.
This is a downstream, CPU-only step; it does not touch the Boltz run.
