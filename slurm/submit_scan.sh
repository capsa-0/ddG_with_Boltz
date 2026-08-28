#!/bin/bash
#==============================================================================
# Submit a full mutational scan as a chained set of SLURM jobs.
#
#   prepare (CPU) -> predict array (GPU, N shards, each self-slimming)
#                  -> slim sweep (CPU, afterany) -> features (CPU)
#
# Differs from submit_all.sh in two ways that matter at scan scale (a 400 aa
# protein is ~7.5k structures, raw z ~80 MB each):
#
#   * Each predict task slims its OWN shard and deletes the raw NPZs immediately, so
#     peak disk is ~one shard's raw, not the whole corpus. The final `slim` is only a
#     cheap sweep for leftovers (a task that died between predicting and slimming),
#     not a monolithic pass over the corpus.
#   * Bad nodes are excluded up front (nodo1/nodo3/nodo5 crash boltz at startup;
#     a single shard landing there fails the array and leaves features stuck on
#     DependencyNeverSatisfied).
#
# Usage:
#   ./slurm/submit_scan.sh <config.yaml> <N_shards> [max_parallel]
# Example (128 shards, 2 GPUs at a time):
#   ./slurm/submit_scan.sh experiment_configs/scan_GLA_human.yaml 128 2
#
# Prefer MANY SHORT shards: a task that dies costs one shard (which requeues),
# not the run. 128 shards over ~7.5k structures is ~60 structures/task.
#==============================================================================
set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> <N_shards> [max_parallel]}"
NSHARDS="${2:?provide the number of predict shards N (e.g. 128)}"
MAXPAR="${3:-2}"   # cap concurrent GPU shards (cluster courtesy); default 2

cd "$(dirname "$0")/.."

# Nodes that break boltz at startup (see CLAUDE.md): bad GPU / CUDA init / ld.so,
# plus GPUs too small for a large protein. Filtered against the live inventory:
# sbatch rejects an ENTIRE submission with "Invalid node name specified" if the
# list names a node that no longer exists (nodo14/nodo15 were decommissioned).
WANT_CPU="nodo1 nodo3 nodo4 nodo5 nodo9"
WANT_GPU="nodo1 nodo3 nodo4 nodo5 nodo9 nodo11 nodo12 sauron"

existing_nodes() {
    scontrol show hostnames "$(sinfo -h -o '%N' | paste -sd, -)" 2>/dev/null
}
filter_nodes() {   # keep only names the cluster actually has
    local have keep=""
    have=$(existing_nodes)
    for n in $1; do
        grep -qx "$n" <<<"$have" && keep="${keep:+$keep,}$n"
    done
    echo "$keep"
}
EXCLUDE_CPU=$(filter_nodes "$WANT_CPU")
EXCLUDE_GPU=$(filter_nodes "$WANT_GPU")
echo "exclude cpu=${EXCLUDE_CPU:-<none>}  gpu=${EXCLUDE_GPU:-<none>}"

echo "config=${CONFIG}  shards=${NSHARDS}  max_parallel=${MAXPAR}"

# 1) prepare (CPU): dataset -> MSAs -> mutated MSAs -> Boltz queries, and warm
#    the Boltz cache serially so the GPU shards only ever read it.
PREP=$(sbatch --parsable --job-name=ddg-prepare ${EXCLUDE_CPU:+--exclude="$EXCLUDE_CPU"} \
        slurm/cpu_step.sbatch "$CONFIG" prepare)
echo "prepare  : job  ${PREP}"

# 2) predict (GPU array): each task predicts its shard, then slims it and drops raw.
PRED=$(sbatch --parsable --job-name=ddg-predict \
        --dependency=afterok:"${PREP}" ${EXCLUDE_GPU:+--exclude="$EXCLUDE_GPU"} \
        --array=0-$((NSHARDS-1))%"${MAXPAR}" \
        slurm/predict_array.sbatch "$CONFIG" "$NSHARDS")
echo "predict  : array ${PRED}  (0-$((NSHARDS-1))%${MAXPAR})"

# 3) slim (CPU): shardless sweep, dependency afterANY the array.
#    afterok would make the whole run hostage to a single transient task failure --
#    even one whose PREDICTIONS completed and were merged (seen in practice: a task
#    lost only its second srun to a SLURM "Error generating job credential", leaving
#    27 good structures unslimmed and features permanently DependencyNeverSatisfied).
#    A shardless slim sweeps up every leftover raw folder, so the run survives it.
SLIM=$(sbatch --parsable --job-name=ddg-slim --dependency=afterany:"${PRED}" \
        ${EXCLUDE_CPU:+--exclude="$EXCLUDE_CPU"} slurm/cpu_step.sbatch "$CONFIG" slim)
echo "slim     : job  ${SLIM}  (afterany: survives an individual shard failure)"

# 4) features (CPU): only if the slim sweep succeeded.
FEAT=$(sbatch --parsable --job-name=ddg-features --dependency=afterok:"${SLIM}" \
        ${EXCLUDE_CPU:+--exclude="$EXCLUDE_CPU"} slurm/cpu_step.sbatch "$CONFIG" features)
echo "features : job  ${FEAT}"

cat <<MSG

Chain submitted. Watch it with:
  squeue -u \$USER -o '%.11i %.14j %.9T %.10M %.6D %R'
  python -m ddg status ${CONFIG}      # via a job, per cluster policy

Predict is resumable (it skips structures already predicted or already in the
slim store), so if the array fails, just re-run this script -- it only redoes
leftover work. If features is left PENDING (DependencyNeverSatisfied), the array
failed: scancel ${FEAT} first, then resubmit.

If a single shard fails, the array still finishes and slim/features still run
(afterany). Check that shard\'s predictions landed, then re-run this script to
redo only what is missing.

When features finishes, score the scan:
  sbatch slurm/scan_predict.sbatch ${CONFIG}
MSG
