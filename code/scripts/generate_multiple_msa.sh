#!/bin/bash

set -e
set -u

FASTA_DIR="$1"
DB_DIR="$2"
OUT_DIR="$3"

mkdir -p "$OUT_DIR"
TMP_DIR="$OUT_DIR/tmp"
mkdir -p "$TMP_DIR"

for fasta_file in "$FASTA_DIR"/*.fasta "$FASTA_DIR"/*.fa; do
    [ -e "$fasta_file" ] || continue

    base_name=$(basename "$fasta_file")
    name="${base_name%.*}"

    echo "Processing $base_name ..."

    QUERY_DB="$TMP_DIR/${name}_queryDB"
    mmseqs createdb "$fasta_file" "$QUERY_DB"

    RESULT_DB="$TMP_DIR/${name}_resultDB"
    mmseqs search "$QUERY_DB" "$DB_DIR" "$RESULT_DB" "$TMP_DIR" \
        --threads 4 --max-seqs 500 --e-value 1e-3


    MSA_DB="$TMP_DIR/${name}_msaDB"
    mmseqs msa "$QUERY_DB" "$DB_DIR" "$RESULT_DB" "$MSA_DB" --threads 4

    MSA_FILE="$OUT_DIR/${name}.a3m"
    mmseqs result2msa "$QUERY_DB" "$DB_DIR" "$MSA_DB" "$MSA_FILE" --format-output "fasta" --threads 4

    echo "MSA saved at $MSA_FILE"
done
