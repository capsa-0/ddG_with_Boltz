# ddG with Boltz

Predict the change in folding free energy (**ΔΔG**) of a single-point protein
mutation from the internal representations of a structure-prediction model —
without ever running its structure head.

Boltz-2 is run in an **embeddings-only** mode (a small patch to the upstream
model, kept in `external/boltz_modified/`). For each mutation we keep the trunk
outputs for the wild-type and the mutant sequence, and regress ΔΔG on features
derived from their difference. Nothing is fine-tuned: the trunk is **frozen**,
and only a small readout (MLP / SVR / ridge) is fitted on top.

The trunk exposes three tensors per structure:

| tensor | shape | what it is |
|---|---|---|
| `s` | L × 384 | single track — one vector per residue |
| `z` | L × L × 128 | pair track — one vector per residue pair |
| `pdistogram` | L × L × bins | predicted distance distribution |

The **pair track is where the stability signal lives.** For a mutation at
0-based position `i`, the feature builder emits blocks of 128 columns each:

- `zdiag` = `mut_z[i,i] − wt_z[i,i]` — the local pair element. This alone
  (128 dims) carries essentially all of the *transferable* signal.
- `zpool` = mean over residues of `mut_z[i,:] − wt_z[i,:]` — a pooled
  *difference*; cancels the per-protein offset, helps in-distribution.
- `wtz` / `mtz` = the pooled rows as *levels*. Useful in-distribution, but they
  import a corpus-specific offset and **hurt cross-corpus transfer**.
- `sdim` = `mut_s[i] − wt_s[i]` — optional, only when `slim.keep_s: true`.

Which blocks to use is a deliberate choice, not a default:
`ddg.evaluation.labels.TRANSFER_BLOCKS` vs `IN_DISTRIBUTION_BLOCKS`.
In-distribution holdout performance **mis-ranks** readouts for cross-corpus use
(see `results/14_biophysical_features/`).

## Where it stands

Every claim below is written up, with configs and figures, in `results/`.
Start at [`results/history.md`](results/history.md) for the narrative and
[`results/README.md`](results/README.md) for the index.

- **It works and it generalizes.** On the Tsuboyama corpus (412 proteins,
  12,359 mutations): random CV pooled r **0.78**, unseen-protein holdout
  **0.70**, 30 %-identity homology holdout **0.765**, per-protein mean r **0.81**
  (`01_generalization`). An MLP matches or beats the tree model on every
  holdout, so this is a property of the representation, not the regressor
  (`06_mlp_generalization`).
- **It transfers to other datasets, at parity with the closest published
  method.** On **S669**, leakage-free (411 variants / 67 proteins, after a
  coverage-free domain audit removed 34.7 % of the corpus): ρ **0.500** /
  r **0.512**, against AFToolkit's 0.453 — paired Δρ +0.054 [−0.017, +0.151], at
  128 dims against 358 and 1/18 the training data. On **FireProt**
  (homology-filtered, ≤500 aa): r **0.65** / ρ **0.66**, and on the 1,173
  variants blind to *both* methods ρ **0.685** against AFToolkit's 0.633
  (`16_aftoolkit_headtohead`, `05_cross_dataset_fireprot`). It also predicts MAVE
  functional fitness better than Rosetta standalone (0.354 vs 0.279), though the
  gain is absorbed by conservation (`15_mave_stability_transfer`).
- **Where it breaks.** It interpolates but does not extrapolate beyond its
  training range (tail r 0.09, `02_stress_extrapolation`); ranking *within* the
  stabilizing tail is the project's open problem — loss reweighting does not fix
  it (`12_error_anatomy`, `13_balanced_loss`). The missing cross-protein term is
  domain shift, not a per-protein property (`11_calibration_gap`).
- MSAs are worth a uniform ~0.08–0.10 r; the structural prior alone still
  reaches 0.70 (`04_no_msa_ablation`).

## Install

```bash
conda env create -f environment.yml
conda activate ddG_with_Boltz
```

This installs the patched Boltz fork (`external/boltz_modified`) and this
package, both editable. Run the CLI **from the repository root** — some paths
(the internal config, the MMseqs helper) resolve relative to it.

## Running an experiment

The pipeline is four steps, in order:

| step | does | writes |
|---|---|---|
| `prepare` | dataset → canonical keys, MSAs, mutated MSAs, Boltz queries | `mutations.csv`, `metadata.csv`, `wt_sequences.fasta`, `msas/`, `queries/` |
| `predict` | run the `boltz` CLI with `--embeddings_only` | `boltz_raw_output/predictions/<id>/embeddings_<id>.npz` |
| `slim` | compact embeddings, drop the raw NPZs | `slim/s<i>.npz` |
| `features` | slim store → the raw-Δz feature table | `features_summary.parquet` |

Everything lands under `data/processed/<experiment_name>/` (gitignored).

```bash
python -m ddg run experiment_configs/tsuboyama_bench_fast.yaml   # all four steps
python -m ddg run <config> --step prepare|predict|slim|features  # one step
python -m ddg status <config>                                    # progress, read from disk
python -m ddg list                                               # all experiments
```

`status` derives state from the files on disk, so it stays correct even if a job
was killed. `predict` is **resumable** — it skips structures already present.

Then the generalization benchmark (tables + figures under
`data/processed/<experiment>/benchmark/`):

```bash
python -m ddg.evaluation --config <config> [--model mlp|svr|ridge|hgb] [--build-clusters]
```

Holdouts: `random`, `protein`, `cluster` (homology, needs `--build-clusters` or
`--cluster-map`), `denovo` (natural ↔ designed), `substitution`,
`source_residue`, `target_residue`, `chemistry`.

### Exhaustive single-protein scans

`ddg.scan` points the predictor at one protein with **no labels** and asks for
every single point mutation (L positions × 19 residues):

```bash
python -m ddg.scan build   --sequence <SEQ> --name <ID>   # writes the CSV + config
./slurm/submit_scan.sh experiment_configs/scan_<ID>.yaml 128 2
python -m ddg.scan predict --config experiment_configs/scan_<ID>.yaml
```

`build` emits an ordinary experiment; nothing downstream knows it is a scan.
See `results/10_gla_scan/` for a worked example (human α-galactosidase A,
7,562 mutations, compared against FoldX).

## Repository layout

```
ddg/                          # the Python package (import root; NOT `src/`)
  cli.py / __main__.py        # `python -m ddg run|status|list`
  pipeline.py                 # step orchestration: prepare -> predict -> slim -> features
  config/                     # ProjectConfig = experiment YAML + internal naming YAML
  datasets/                   # input adapters -> unified MutationSample; prepare step
  feature_extraction/         # MSAs, mutated MSAs, Boltz queries; runs the boltz CLI
  storage/                    # slim store: compact per-mutation embedding slices
  features/build_features.py  # features step -> features_summary.parquet
  evaluation/                 # holdout benchmark suite: splits, labels, models, metrics, plots
  scan/                       # exhaustive single-protein scans
  state/                      # on-disk run state; powers `ddg status`

external/boltz_modified/      # PATCHED Boltz (editable) — see its changes.md
external/mmseqs.py            # ColabFold-style MMseqs2 server client
experiment_configs/*.yaml     # per-run parameters
slurm/                        # SLURM submit scripts
data/raw/                     # input CSVs (committed)
data/processed/               # per-experiment outputs (gitignored)
ddg_datasets/                 # dataset-cleaning working area (mostly untracked)
results/                      # one committed folder per result
theory/                       # papers and design notes
tests/                        # pytest
TODO.md                       # planning / working notes
```

## Configuration

Two YAMLs are merged by `ProjectConfig`:

- **`experiment_configs/<name>.yaml`** — per-run choices: `raw_data_path`,
  `dataset_type` (`fireprot` | `dms` | `minimal`), MSA strategy,
  `max_msa_sequences`, `boltz_flags`, `slim.keep_s`, `feature.blocks`.
- **`ddg/config/internal_config.yaml`** — fixed directory and filename
  conventions. Rarely edited.

Prefer adding a parameter to the experiment YAML over hard-coding a path or a
flag. Note `overwrite: true` **deletes** the whole processed directory for that
experiment before running — set it deliberately.

## Conventions & gotchas

- **The Boltz install is patched.** The only functional change is the
  `--embeddings_only` flag (`external/boltz_modified/changes.md`). The pin is
  intentional; upgrading Boltz means re-applying that patch. Do not assume
  upstream Boltz behaves identically.
- **The ΔΔG column name differs by adapter.** FireProt expects `ddG`; DMS and
  minimal expect `ddg`. `MutationSample.ddg` is lower-case everywhere downstream.
- **Mutation strings are `<WT><1-based-pos><MUT>`** (e.g. `P8A`). The 0-based
  index into the embedding assumes the Boltz query sequence is 1:1 with the
  dataset WT sequence; `prepare` validates `sequence[pos-1] == wt_aa`.
- **The MSA mutation strategy is a scientific variable, not formatting.**
  `mutate_across_msa` edits the mutated column in *every* MSA row;
  `mutate_first_row` edits only the query row. They change what evolutionary
  signal Boltz sees.
- **MMseqs2 clustering is not a sufficient leakage check here.** The training
  corpus is Megascale domains (~70 aa) excised from real proteins; a benchmark
  protein of several hundred residues can contain one verbatim and still never
  reach the 80 % coverage `easy-cluster` needs. This hid contamination in 34.7 %
  of S669. Screen transfer corpora with the coverage-free local alignment in
  `results/16_aftoolkit_headtohead/domain_leakage_audit.py` as well.
- **MSAs are reusable.** They are keyed by `wt_id` (`{wt_id}.a3m`) and identical
  across corpora sharing the same WT proteins. When the ColabFold MMseqs2 server
  rate-limits, copy them from a finished experiment's `msas/` and re-run
  `prepare` — it skips any MSA already on disk.
- **Only the last recycling step is kept.** When `s`/`z` come back with an extra
  leading dimension, `ddg.storage.slim` collapses it to the final step.
- New dataset formats go behind a new adapter in `ddg/datasets/`, registered in
  `load_input_dataset.py` — don't special-case formats deeper in the pipeline.
- Comments and logs in `ddg/` are in English.
- Dataset *cleaning* (building the corpora in `data/raw/`) happens in
  `ddg_datasets/`, which is a local working area and mostly untracked; only
  `build_benchmark_corpus.py` is committed.

## Running on the cluster

The real runs happen on a SLURM cluster. **Never run compute on the login
node** — `python`, `boltz`, feature extraction and analysis all go through a
job; `git`, `squeue`, `sacct`, `ls` and friends are fine.

```bash
./slurm/submit_all.sh <config> <N_shards> [max_parallel]   # the whole chain, afterok-linked
sbatch slurm/cpu_step.sbatch <config> prepare|slim|features
sbatch --array=0-$((N-1))%M slurm/predict_array.sbatch <config> N
sbatch slurm/eval.sbatch <config> [hgb|svr|ridge|mlp]      # note: defaults to hgb, not mlp
```

`submit_all.sh` chains four jobs — prepare (CPU) → predict (GPU array) →
slim (CPU) → features (CPU) — each starting only if the previous succeeded.
Watch for `PENDING (DependencyNeverSatisfied)`: that means an upstream step
already failed and the job will never run, so `scancel` it, fix the cause, and
re-submit (predict is resumable, so it only redoes the missing structures).

Two things that were learned the hard way and are baked into the scripts:

- **Many short shards, not few long jobs.** A long job that dies near the end
  wastes all of it *and* stalls the `afterok` chain. Use a high shard count
  (64+) with a low GPU concurrency cap.
- **`predict_array.sbatch` slims each shard immediately** (`delete_raw`), so
  only ~one shard's raw embeddings exist at a time. Raw `z` for a ≥300 aa
  protein is ~30 MB; without this a single corpus can peak at ~180 GB.

Job scripts activate conda via `source /home/shared/load-conda`. Some nodes are
known-bad for `boltz predict` and are excluded by the scripts; if failures
cluster on one node while others succeed concurrently, it is the node.

## Tests

```bash
pytest tests/test_prepare.py tests/test_sharding.py tests/test_cli_status.py
```

These cover mutation parsing and WT-identity validation in `prepare`, the
round-robin predict sharding, and the on-disk `ddg status` view.

The other three (`test_boltz_dataset_keys.py`, `test_slim_dataset.py`,
`test_slim_equivalence.py`) are **stale**: they import
`ddg.datasets.boltz_dataset`, the raw-NPZ reader removed in the raw-Δz refactor
(20f5352), so a bare `pytest tests/` fails at collection. They need porting to
`ddg.storage.slim_store` or deleting — see `TODO.md`.

## Reading further

- [`results/history.md`](results/history.md) — the narrative thread: why raw Δz,
  how generalization was established, where it breaks.
- [`results/README.md`](results/README.md) — index of all 16 results with
  headline numbers.
- [`results/guidelines.md`](results/guidelines.md) — what a result folder must
  contain, and the `status.md` logging rule.
- [`TODO.md`](TODO.md) — planning and working notes.
- [`theory/sota_2026.md`](theory/sota_2026.md) — the surrounding literature.
