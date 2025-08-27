#!/bin/bash


INPUT_DIR="$1"
OUTPUT_DIR="$2"

mkdir -p "$OUTPUT_DIR"

for file in "$INPUT_DIR"/*.a3m; do
    filename=$(basename "$file")
    
    base="${filename%.a3m}"
    
    uniprot_id="${base%%_*}"
    
    mutations="${base#*_}"

    IFS='_' read -ra muts <<< "$mutations"
    for mut in "${muts[@]}"; do
        output_file="$OUTPUT_DIR/${uniprot_id}_${mut}.a3m"
        
        python scripts/mut_msa.py "$file" "$output_file" "$mut"
    done
done
