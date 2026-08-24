#!/bin/bash
#==============================================================================
# Run the generalization benchmark (ddg.evaluation) over features_summary.parquet
# as a CPU job. Writes tables + figures under data/processed/<exp>/benchmark/.
#
# Usage:
#   sbatch slurm/eval.sbatch <config.yaml> [model]        # model: hgb (default) | svr | ridge | mlp
#   sbatch slurm/eval.sbatch experiment_configs/tsuboyama_bench_fast.yaml hgb
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
MODEL="${2:-hgb}"

echo "EVAL  job ${SLURM_JOB_ID}  node ${SLURM_JOB_NODELIST}  config ${CONFIG}  model ${MODEL}"

cd "$PROJECT_ROOT"
# Cluster conda is loaded through the shared loader (admin-mandated 2026-08-24;
# the old `conda shell.bash hook` / `source activate` pair no longer resolves).
source /home/shared/load-conda
conda activate ddG_with_Boltz

# Extra args after the model are forwarded (e.g. --drop-s for the s-ablation).
srun python -m ddg.evaluation --config "$CONFIG" --model "$MODEL" "${@:3}"
