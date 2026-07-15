#!/bin/bash
#==============================================================================
# Run a single CPU-only pipeline step (prepare / slim / features) as a job.
# Only `predict` needs a GPU (use slurm/predict_array.sbatch for that).
#
# Usage:
#   sbatch slurm/cpu_step.sbatch <config.yaml> prepare
#   sbatch slurm/cpu_step.sbatch <config.yaml> slim
#   sbatch slurm/cpu_step.sbatch <config.yaml> features
#==============================================================================
#SBATCH --job-name=ddg-cpu
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e

# --- Set this to your project root on the cluster ---
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
# ----------------------------------------------------

CONFIG="${1:?Usage: sbatch slurm/cpu_step.sbatch <config.yaml> <step>}"
STEP="${2:?provide a step: prepare | slim | features}"

echo "CPU step '${STEP}'  job ${SLURM_JOB_ID}  node ${SLURM_JOB_NODELIST}  config ${CONFIG}"

cd "$PROJECT_ROOT"
eval "$(conda shell.bash hook)"
source activate ddG_with_Boltz

srun python -m ddg run "$CONFIG" --step "$STEP"
