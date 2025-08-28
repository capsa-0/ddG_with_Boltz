#!/bin/bash

set -e 

cd "$(dirname "$0")/.."

DATASET_TYPE="$1"
RAW_DB_PATH="$2"

DATASET_NAME="$(basename "$RAW_DB_PATH" | sed 's/\.[^.]*$//')"
OUTPUT_DIR="data/processed/$DATASET_NAME/"

echo "Cleaning previous outputs..."
python clean.py


echo "Loading dataset..."
python src/ddg_predictor/data_prep/parse_dataset/load_dataset.py \
    --dataset_type "$DATASET_TYPE" \
    --raw_path "$RAW_DB_PATH" \
    --output_dir "$OUTPUT_DIR" 

OUTPUT_FASTA="$OUTPUT_DIR/wt_sequences.fasta"
OUTPUT_MUT_DATA="$OUTPUT_DIR/mut_data.csv"

echo "Generating MSAs..."
python src/ddg_predictor/data_prep/get_msas/msa_stack.py \
    --input_fasta "$OUTPUT_FASTA" \
    --output_dir "$OUTPUT_DIR"\
    --mutations_csv "$OUTPUT_MUT_DATA"

MSAS_DIR="${OUTPUT_DIR}msas/"

python src/ddg_predictor/data_prep/to_boltz_query/m3a_to_yaml.py \
    "$MSAS_DIR" \
    "$OUTPUT_DIR" \
    "config/boltz_query_template.yaml"

