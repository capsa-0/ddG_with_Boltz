#!/bin/bash
#==============================================================================
# Length-ceiling probe: run ONE Boltz query per array task, so an OOM on a long
# chain costs that structure only instead of killing the whole run (the normal
# predict step hands every pending query to a single boltz process).
#
#   sbatch --array=0-13%1 -p gpu --gres=gpu:1 slurm/probe_length.sbatch <config> 14
#   sbatch --array=0-13%2 -p cpu --mem=60G    slurm/probe_length.sbatch <config> 14
#
# Records the GPU model/compute-capability and peak VRAM, so the result says
# *which* card the ceiling belongs to. Peak host RAM comes from `sacct -o MaxRSS`.
#==============================================================================
#SBATCH --job-name=ddg-probe
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH -e logs/slurm-%A_%a.err
#SBATCH -o logs/slurm-%A_%a.out

PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
CONFIG="${1:?Usage: sbatch slurm/probe_length.sbatch <config.yaml> <N>}"
N="${2:?provide the shard count}"

echo "probe shard ${SLURM_ARRAY_TASK_ID}/${N}  job ${SLURM_JOB_ID}  node $(hostname)"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader || \
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    # sample VRAM every 2 s; the max is reported at the end
    ( while true; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null
        sleep 2
      done ) > "/tmp/vram_${SLURM_JOB_ID}.txt" 2>/dev/null &
    POLLER=$!
fi

cd "$PROJECT_ROOT"
source /home/shared/load-conda
conda activate ddG_with_Boltz

srun python -m ddg run "$CONFIG" --step predict --shard "${SLURM_ARRAY_TASK_ID}/${N}"
STATUS=$?

if [ -n "${POLLER:-}" ]; then
    kill "$POLLER" 2>/dev/null
    echo "peak VRAM (MiB): $(sort -n "/tmp/vram_${SLURM_JOB_ID}.txt" 2>/dev/null | tail -1)"
    rm -f "/tmp/vram_${SLURM_JOB_ID}.txt"
fi
echo "exit status: $STATUS"
exit $STATUS
