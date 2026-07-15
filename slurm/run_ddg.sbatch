#!/bin/bash
#==============================================================================
# Run the ddg pipeline for one experiment via the `ddg` CLI.
#
# Usage:  sbatch slurm/run_ddg.sbatch experiment_configs/tsuboyama_test.yaml
# Then :  ddg status experiment_configs/tsuboyama_test.yaml   (from the login node)
#==============================================================================
#SBATCH --job-name=ddg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
# GPU partition has no time limit -> no --time.
# Exclude nodes whose GPU is too small/old for Boltz-2 (verify nodo13 yourself).
#SBATCH --exclude=nodo11,nodo12,nodo14,nodo15,sauron
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

set -e

# --- Set this to your project root on the cluster ---
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
# ----------------------------------------------------

CONFIG="${1:?Usage: sbatch slurm/run_ddg.sbatch <experiment_config.yaml>}"

echo "================================================================"
echo "Job ${SLURM_JOB_ID} on ${SLURM_JOB_NODELIST}  config=${CONFIG}"
date +"Start: %F - %T"
echo "================================================================"

cd "$PROJECT_ROOT"
eval "$(conda shell.bash hook)"
source activate ddG_with_Boltz

# Runs all steps in order: prepare -> predict -> slim -> features.
# (You can also run a single step:  ddg run "$CONFIG" --step predict)
srun ddg run "$CONFIG"

echo "================================================================"
date +"End: %F - %T"
sacct --format=JobID,State,Partition,Elapsed,MaxRSS,NodeList --units=M -j "$SLURM_JOB_ID" || true
echo "================================================================"
