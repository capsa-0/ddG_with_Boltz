#!/bin/bash
#==============================================================================
# Sharded Boltz prediction across several GPUs (SLURM array job).
#
# Flow:
#   1) generate queries once (CPU + network for MSAs):
#        python -m ddg run <config.yaml> --step prepare
#   2) predict in parallel across N shards, at most 3 GPUs at a time (%3):
#        sbatch --array=0-$((N-1))%3 slurm/predict_array.sbatch <config.yaml> N
#   3) after the array finishes, finalize once (login node or a small job):
#        python -m ddg run <config.yaml> --step slim
#        python -m ddg run <config.yaml> --step features
#   Monitor any time:  python -m ddg status <config.yaml>
#==============================================================================
#SBATCH --job-name=ddg-predict
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# No --time (gpu partition is unlimited). Exclude too-small/old GPUs and
# nodo3 (broken dynamic linker: boltz aborts with ld.so assertion / exit 127).
#SBATCH --exclude=nodo3,nodo11,nodo12,nodo14,nodo15,sauron
# Auto-requeue if the node fails/preempts (predict is resumable: on rerun it
# skips queries already predicted, so requeue only redoes leftover work).
#SBATCH --requeue
#SBATCH -e logs/slurm-%A_%a.err
#SBATCH -o logs/slurm-%A_%a.out

set -e

# --- Set this to your project root on the cluster ---
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
# ----------------------------------------------------

CONFIG="${1:?Usage: sbatch --array=0-N-1%3 slurm/predict_array.sbatch <config.yaml> <N_shards>}"
NSHARDS="${2:?provide the number of shards N as the 2nd argument (must match the --array range)}"

echo "Predict shard ${SLURM_ARRAY_TASK_ID}/${NSHARDS}  job ${SLURM_ARRAY_JOB_ID}  node ${SLURM_JOB_NODELIST}  config ${CONFIG}"

cd "$PROJECT_ROOT"
eval "$(conda shell.bash hook)"
source activate ddG_with_Boltz

srun python -m ddg run "$CONFIG" --step predict --shard "${SLURM_ARRAY_TASK_ID}/${NSHARDS}"

# Incremental slim: compact THIS shard's raw predictions into s<i>.npz and delete
# the raw NPZs right away (delete_raw), so full-size raw embeddings never pile up
# across the whole array. Bounds peak disk — critical for large (>=300 aa) proteins
# whose raw z is ~30 MB each. The final slim step (if any) is then just a safety net
# for leftover raw; features reads all slim shards (s0000.npz, ...).
srun python -m ddg run "$CONFIG" --step slim --shard "${SLURM_ARRAY_TASK_ID}/${NSHARDS}"
