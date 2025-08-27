#!/bin/bash
set -euo pipefail

# -------------------------------------------------------------
# Full pipeline to go from Excel with sequences to MSAs using MMseqs2 and Uniref50
# -------------------------------------------------------------

# 1 Extracting sequences from Excel to multifasta
echo ">>> Extracting sequences from raw_data/S11304.xlsx..."
python scripts/extract_sequences.py raw_data/S11304.xlsx intermediate/wt_sequences.fa

# 3 Generating Uniref50 MMseqs2 database
echo ">>> Generating Uniref50 MMseqs2 database..."
mkdir -p intermediate/uniref50_lite_db
mmseqs createdb raw_data/uniref50_sampled.fasta.gz intermediate/uniref50_lite_db/uniref_db

# 4 Generating MSAs for each sequence
echo ">>> Generating MSAs for each sequence..."
mkdir -p msas
scripts/multifasta_to_msas.sh intermediate/wt_sequences.fa intermediate/uniref50_lite_db/uniref_db msas

# Remove unnecessary MMseqs2 database files
echo ">>> Cleaning up unnecessary database files..."
rm -rf msas/*.index msas/*.dbtype intermediate

# 5 Generating mutated MSAs
echo ">>> Generating mutated MSAs..."
mkdir -p all_msas
scripts/mut_msa_directory.sh msas all_msas

# 6 Cleanup: renaming files and moving to final folder
echo ">>> Renaming and moving final MSAs to 'msas' folder..."
scripts/rename_files.sh msas
mv msas/* all_msas/
rm -rf msas

# 7 Generating YAML files for each MSA
echo ">>> Generating yaml files for each MSA"
mkdir -p yaml_files
python scripts/get_yaml.py all_msas yaml_files raw_data/boltz_minimum_query.yaml

# -------------------------------------------------------------
echo "Done - All YAML files are in 'yaml_files'"
