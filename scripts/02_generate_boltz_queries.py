# scripts/02_generate_boltz_queries.py

import yaml
import os
import logging
from ddg_predictor.data_processing import msa_handler

from ddg_predictor.data_processing.loaders import load_config
load_config('params.yaml')

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Runs the full MSA preparation pipeline: generation, mutation, and processing.
    """
    logging.info("--- STEP 2: Preparing All MSAs ---")
    
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)['data_processing']

    # --- Construct input file paths ---
    fasta_path = os.path.join(config['output_dir'], config['wt_fasta_filename'])
    mutations_csv_path = os.path.join(config['output_dir'], config['mutations_csv_filename'])

    # --- Verify that input files exist ---
    for path in [fasta_path, mutations_csv_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found at '{path}'. Please run STEP 1 first.")

    # --- Execute MSA logic ---
    # 1. Generate MSAs for Wild-Type sequences
    msa_handler.generate_wt_msas(
        fasta_path=fasta_path,
        output_dir=config['msa_output_dir']
    )

    # 2. Apply mutations to create mutant MSAs
    msa_handler.apply_mutations_to_msas(
        msa_dir=config['msa_output_dir'],
        mutations_csv_path=mutations_csv_path
    )

    # 3. Truncate all MSAs and convert them to YAML queries
    msa_handler.process_msas(
        msa_dir=config['msa_output_dir'],
        yaml_output_dir=config['queries_dir'],
        max_sequences=config.get('max_msa_sequences')
    )

    logging.info("--- STEP 2: Finished ---")

if __name__ == '__main__':
    try:
        main()
    except FileNotFoundError as e:
        logging.error(e)
        exit(1) # Exit with a non-zero code to signal failure
    except Exception as e:
        logging.error(f"An unexpected error occurred in Step 2: {e}", exc_info=True)
        exit(1)