#!/bin/bash
#SBATCH --job-name=s669ext-feat
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH -e logs/slurm-%j.err
#SBATCH -o logs/slurm-%j.out
set -e
cd /grupos/Marce/estructural/ddG_with_Boltz/ddG_with_Boltz
source /home/shared/load-conda
conda activate ddG_with_Boltz
# `python results/.../script.py` puts the SCRIPT's directory on sys.path, not the project
# root, so `import ddg` fails. The pipeline itself never hits this because it runs via
# `python -m ddg`, which does add the cwd.
export PYTHONPATH="$PWD:$PYTHONPATH"
echo "== 1. ablation features for s669_ext =="
srun python results/07_feature_symmetry_ablation/build_ablation_features.py data/processed/s669_ext
echo "== 2. bio features (contact-weighted / far-shell / substitution identity) =="
srun python results/14_biophysical_features/build_bio_features.py --exp s669_ext
echo "== 3. merge into the 629-variant corpus =="
srun python results/16_aftoolkit_headtohead/build_s669_full.py
echo "== 4. transfer the Tsuboyama-trained models onto it =="
srun python results/14_biophysical_features/run_ablation.py \
     --configs diag dz dz_cw cw base far onehot --no-augment \
     --transfer s669_full --out results_s669_full.csv
echo "== done =="
