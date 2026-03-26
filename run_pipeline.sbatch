#!/bin/bash

#==============================================================================
# SLURM JOB CONFIGURATION
#==============================================================================
#SBATCH --job-name=pipeline_ddg_boltz_1   # Job name
#SBATCH --partition=gpu            # Partition (queue) to submit the job to
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # Request 8 CPUs
#SBATCH --mem=32G                  # Request 32 GB of RAM
#SBATCH --nodelist=nodo8           # Execute on a specific node (optional)
#SBATCH -e slurm-%j.err            # File for standard error output
#SBATCH -o slurm-%j.out            # File for standard output

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================
# --- Modify this path ---
# Set the absolute path to your project's root directory.
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz"
# ------------------------

# Exit the script immediately if any command fails (good practice)
set -e

#==============================================================================
# LOGGING AND PREPARATION
#==============================================================================
echo "================================================================"
echo "Job Name: ${SLURM_JOB_NAME}"
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: ${SLURM_JOB_NODELIST}"
echo "Project Root: ${PROJECT_ROOT}"
echo "----------------------------------------------------------------"
date +"Start: %F - %T"
echo "================================================================"
echo

# Navigate to the project directory to ensure relative paths work
cd "$PROJECT_ROOT"

# Activate the Conda environment
echo "Activating Conda environment: ddG_with_Boltz"
eval "$(conda shell.bash hook)"
source activate ddG_with_Boltz
echo "Environment activated."
echo

#==============================================================================
# PIPELINE EXECUTION
#==============================================================================
echo "Starting pipeline execution..."

srun python -m src.main --config experiment_configs/experiment2.yaml

echo
echo ">>> Pipeline completed successfully. <<<"

#==============================================================================
# COMPLETION AND RESOURCE LOGGING
#==============================================================================
echo
echo "================================================================"
date +"End: %F - %T"
echo "----------------------------------------------------------------"
echo "Resource Usage:"
# Display a summary of the job's resource usage upon completion
sacct --format=JobID,Submit,Start,End,State,Partition,ReqTRES%30,CPUTime,MaxRSS,NodeList --units=M -j $SLURM_JOB_ID
echo "================================================================"
