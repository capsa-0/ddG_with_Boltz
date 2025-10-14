#!/bin/bash


#SBATCH --job-name=ddg_boltz_pipeline_1

#SBATCH --partition=gpu

#SBATCH --gres=gpu:1

#SBATCH --cpus-per-task=8

#SBATCH --mem=32G

#SBATCH -e slurm-%j.err

#SBATCH -o slurm-%j.out

#SBATCH --nodelist=nodo9    # Elegir nodo


# --- Configuration ---

# Set this to the absolute path of your boltz project directory.

PROJECT_ROOT="/grupos/Marce/estructural/ddG_with_Boltz"

RESULTS_DIR_5="/grupos/Marce/estructural/ddG_with_Boltz/results"

# ---------------------


# Create the results directory if it doesn't exist

mkdir -p "$RESULTS_DIR_5"


# Log job information

echo "Job Name: ${SLURM_JOB_NAME}"

echo "Job ID: ${SLURM_JOB_ID}"

echo "Partition: ${SLURM_JOB_PARTITION}"

echo "Node(s): ${SLURM_JOB_NODELIST}"

echo "CPUs per task: ${SLURM_CPUS_PER_TASK}"

echo "Project Root: ${PROJECT_ROOT}"

echo "Submission Dir: ${SLURM_SUBMIT_DIR}"

echo "Results Dir: ${RESULTS_DIR_5}"

echo


date +"Start: %F - %T"

echo "--------------------------------------------------------------------------------"


# Navigate to the project directory to ensure correct module resolution

cd "$PROJECT_ROOT"


# Activate conda environment

eval "$(conda shell.bash hook)"

source activate boltz_ddg


# Execute the command

echo "Running command:"

echo "python -m exploration.run_msa_collection_experiment \\

    --query_sequence \"MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG\" \\

    --msa_dir \"$PROJECT_ROOT/MSA/all_runs_a3m\" \\

    --msa_pattern \"*.a3m\" \\

    --exp_dir \"$RESULTS_DIR_5\" \\

    --parallel_jobs 1 \\

    --timeout 300 \\

    --resume"


echo

srun python -m exploration.run_msa_collection_experiment \

    --query_sequence "MRIILLGAPGAGKGTQAQFIMEKYGIPQISTGDMLRAAVKSGSELGKQAKDIMDAGKLVTDELVIALVKERIAQEDCRNGFLLDGFPRTIPQADAMKEAGINVDYVLEFDVPDELIVDRIVGRRVHAPSGRVYHVKFNPPKVEGKDDVTGEELTTRKDDQEETVRKRLVEYHQMTAPLIGYYSKEAEAGNTKYAKVDGTKPVAEVRADLEKILG" \

    --msa_dir "$PROJECT_ROOT/MSA/all_runs_a3m" \

    --msa_pattern "*.a3m" \

    --exp_dir "$RESULTS_DIR_5" \

    --parallel_jobs 1 \

    --timeout 300 \

    --resume


echo "--------------------------------------------------------------------------------"

echo "Collecting all predicted structure files..."


STRUCTURES_DIR="$RESULTS_DIR_5/all_structures"

mkdir -p "$STRUCTURES_DIR"

find "$RESULTS_DIR_5" -type f -name "*.pdb" -exec cp -t "$STRUCTURES_DIR" {} +


echo "All PDB files have been copied to: $STRUCTURES_DIR"

echo "--------------------------------------------------------------------------------"

date +"End: %F - %T"

echo


# Log resource usage information

echo "Resource Usage:"

sacct --format=JobID,Submit,Start,End,State,Partition,ReqTRES%30,CPUTime,MaxRSS,NodeList --units=M -j $SLURM_JOBID


