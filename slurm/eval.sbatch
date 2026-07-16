#!/bin/bash
#==============================================================================
# Run the generalization benchmark (ddg.evaluation) over features_summary.parquet
# as a CPU job. Writes tables + figures under data/processed/<exp>/benchmark/.
#
# Usage:
#   sbatch slurm/eval.sbatch <config.yaml> [model]        # model: svr (default) | ridge | mlp
#   sbatch slurm/eval.sbatch experiment_configs/tsuboyama_bench_fast.yaml svr
#==============================================================================
#SBATCH --job-name=ddg-eval
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

set -e

# --- Set this to your project root on the cluster ---
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
# ----------------------------------------------------

CONFIG="${1:?Usage: sbatch slurm/eval.sbatch <config.yaml> [model]}"
MODEL="${2:-svr}"

echo "EVAL  job ${SLURM_JOB_ID}  node ${SLURM_JOB_NODELIST}  config ${CONFIG}  model ${MODEL}"

cd "$PROJECT_ROOT"
eval "$(conda shell.bash hook)"
source activate ddG_with_Boltz

srun python -m ddg.evaluation --config "$CONFIG" --model "$MODEL"
