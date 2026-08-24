#!/bin/bash
#==============================================================================
# Score a finished mutational scan: fit the ΔΔG regressor on the labelled corpora
# (regimes A/B/D) and predict every mutation of the scanned protein.
#
# CPU-only -- the GPU work is over by this point; this just trains 15 small MLPs
# and applies them. Writes tables + figures under data/processed/<exp>/scan/.
#
# Usage:
#   sbatch slurm/scan_predict.sbatch <scan_config.yaml> [extra args...]
#   sbatch slurm/scan_predict.sbatch experiment_configs/scan_GLA_human.yaml
#==============================================================================
#SBATCH --job-name=ddg-scan
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out

set -e

# --- Set this to your project root on the cluster ---
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
# ----------------------------------------------------

CONFIG="${1:?Usage: sbatch slurm/scan_predict.sbatch <scan_config.yaml>}"

echo "SCAN predict  job ${SLURM_JOB_ID}  node ${SLURM_JOB_NODELIST}  config ${CONFIG}"

cd "$PROJECT_ROOT"
# Cluster conda is loaded through the shared loader (admin-mandated 2026-08-24;
# the old `conda shell.bash hook` / `source activate` pair no longer resolves).
source /home/shared/load-conda
conda activate ddG_with_Boltz

srun python -m ddg.scan predict --config "$CONFIG" "${@:2}"
