#!/bin/bash
#==============================================================================
# Submit the whole ddg pipeline as four dependent SLURM jobs, chained so each
# step starts only if the previous one SUCCEEDED (afterok):
#
#   prepare (CPU)  ->  predict (GPU array, N shards)  ->  slim (CPU)  ->  features (CPU)
#
# The parallelism lives in the predict array (N shards, at most MAXPAR GPUs at
# once); prepare/slim/features are single CPU jobs. You run this ONCE from the
# login node -- it only issues `sbatch`, it does no compute itself.
#
# Usage:
#   ./slurm/submit_all.sh <config.yaml> <N_shards> [max_parallel]
# Example (8 shards, 3 GPUs at a time):
#   ./slurm/submit_all.sh experiment_configs/tsuboyama_e1_first_row.yaml 8 3
#
# Monitor:   squeue -u $USER
#            python -m ddg status <config.yaml>   (via a job, per cluster policy)
#==============================================================================
set -euo pipefail

CONFIG="${1:?Usage: $0 <config.yaml> <N_shards> [max_parallel]}"
NSHARDS="${2:?provide the number of predict shards N (e.g. 8)}"
MAXPAR="${3:-3}"   # cap concurrent GPU shards (cluster courtesy); default 3

# Resolve to the project root so logs/ and relative paths behave.
cd "$(dirname "$0")/.."

echo "config=${CONFIG}  shards=${NSHARDS}  max_parallel=${MAXPAR}"

# 1) prepare (CPU): build queries + warm the Boltz cache serially.
PREP=$(sbatch --parsable slurm/cpu_step.sbatch "$CONFIG" prepare)
echo "prepare  : job  ${PREP}"

# 2) predict (GPU array): only if prepare succeeded. Parallelism is here.
PRED=$(sbatch --parsable \
        --dependency=afterok:"${PREP}" \
        --array=0-$((NSHARDS-1))%"${MAXPAR}" \
        slurm/predict_array.sbatch "$CONFIG" "$NSHARDS")
echo "predict  : array ${PRED}  (0-$((NSHARDS-1))%${MAXPAR})"

# 3) slim (CPU): only after the WHOLE predict array succeeds.
SLIM=$(sbatch --parsable --dependency=afterok:"${PRED}" \
        slurm/cpu_step.sbatch "$CONFIG" slim)
echo "slim     : job  ${SLIM}"

# 4) features (CPU): only after slim succeeds.
FEAT=$(sbatch --parsable --dependency=afterok:"${SLIM}" \
        slurm/cpu_step.sbatch "$CONFIG" features)
echo "features : job  ${FEAT}"

echo
echo "Chain submitted. Watch it with:  squeue -u \$USER"
echo
echo "If a predict shard dies on a bad node, the whole array is marked failed and"
echo "slim/features stay PENDING with reason (DependencyNeverSatisfied). Recover by:"
echo "  scancel ${SLIM} ${FEAT}"
echo "  # predict is resumable -- rerun only the leftover work, then re-chain:"
echo "  ./slurm/submit_all.sh ${CONFIG} ${NSHARDS} ${MAXPAR}   # skips finished predictions"
