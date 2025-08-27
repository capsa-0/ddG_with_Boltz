#!/bin/bash
set -euo pipefail

QUERY_FASTA=$1
TARGET_DB=$2
OUTPUT_DIR=$3

TMP_DIR="tmp_msa"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$TMP_DIR"


awk -v tmp="$TMP_DIR" '
    /^>/ {
        if (seq) {
            file = tmp "/" header ".fasta"
            print ">" header > file     
            print seq >> file
        }

        split($0, a, " ")
        gsub(/^>/,"",a[1])
        header=a[1]
        seq=""
        next
    }
    { seq=seq $0 }
    END {
        file = tmp "/" header ".fasta"
        print ">" header > file       
        print seq >> file
    }
' "$QUERY_FASTA"


for query_file in "$TMP_DIR"/*.fasta; do
    uniprot_id=$(basename "$query_file" .fasta)
    echo ">>> Processing $uniprot_id"

    QUERY_DB_TMP="$TMP_DIR/${uniprot_id}_db"
    mmseqs createdb "$query_file" "$QUERY_DB_TMP"

    RESULT_DB="$TMP_DIR/result_${uniprot_id}"
    MSA_OUT="$OUTPUT_DIR/${uniprot_id}.a3m"


    mmseqs search "$QUERY_DB_TMP" "$TARGET_DB" "$RESULT_DB" "$TMP_DIR"

    mmseqs result2msa "$QUERY_DB_TMP" "$TARGET_DB" "$RESULT_DB" "$MSA_OUT" --msa-format-mode 5
done

rm -rf "$TMP_DIR"

