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


# -------------------------------------------------------------
echo "Done - All MSAs are in the 'msas' folder"
