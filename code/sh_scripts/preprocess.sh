#!/bin/bash

set -e 

cd "$(dirname "$0")/.."

DATASET_TYPE="$1"
RAW_DB_PATH="$2"

DATASET_NAME="$(basename "$RAW_DB_PATH" | sed 's/\.[^.]*$//')"
OUTPUT_DIR="data/processed/$DATASET_NAME/"

echo "Loading dataset"
python src/ddg_predictor/data_prep/parse/load_dataset.py \
    --dataset_type "$DATASET_TYPE" \
    --raw_path "$RAW_DB_PATH" \
    --output_dir "$OUTPUT_DIR" 

OUTPUT_FASTA="$OUTPUT_DIR/wt_sequences.fasta"

echo "Generating MSAs for wild-type sequences"
python src/ddg_predictor/data_prep/msa/generate_wt_msas.py \
    --input_fasta "$OUTPUT_FASTA" \
    --output_dir "$OUTPUT_DIR"
