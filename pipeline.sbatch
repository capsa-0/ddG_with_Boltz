#!/bin/bash

#==============================================================================
# SLURM JOB CONFIGURATION
#==============================================================================
#SBATCH --job-name=full_pipeline   # Job name
#SBATCH --partition=gpu            # Partition (queue) to submit the job to
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --cpus-per-task=8          # Request 8 CPUs
#SBATCH --mem=32G                  # Request 32 GB of RAM
#SBATCH --nodelist=nodo9           # Execute on a specific node (optional)
#SBATCH -e slurm-%j.err            # File for standard error output
#SBATCH -o slurm-%j.out            # File for standard output

#==============================================================================
# ENVIRONMENT SETUP
#==============================================================================
# --- Modify this path ---
# Set the absolute path to your project's root directory.
PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz"
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
echo "Activating Conda environment: boltz_ddg"
eval "$(conda shell.bash hook)"
source activate boltz_ddg
echo "Environment activated."
echo

#==============================================================================
# PIPELINE EXECUTION
#==============================================================================
echo "Starting pipeline execution..."

python3 scripts/01_prepare_dataset.py
echo "Step 1/6 completed: prepare_dataset.py"

python3 scripts/02_generate_boltz_queries.py
echo "Step 2/6 completed: generate_boltz_queries.py"

python3 scripts/03_extract_features.py
echo "Step 3/6 completed: extract_features.py"

python3 scripts/04_clean_outputs.py
echo "Step 4/6 completed: clean_outputs.py"

python3 scripts/05_process_features.py
echo "Step 5/6 completed: process_features.py"

python3 scripts/06_analyze_features.py
echo "Step 6/6 completed: analyze_features.py"

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
