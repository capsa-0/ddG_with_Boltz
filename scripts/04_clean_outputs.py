# scripts/04_clean_outputs.py

import yaml
import logging
from ddg_predictor.feature_extraction.output_cleaner import cleanup_boltz_output

from ddg_predictor.data_processing.loaders import load_config
load_config('params.yaml')

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Cleans and reorganizes the raw output directory from Boltz and archives
    the config and mutation files used for the run.
    """
    logging.info("--- STEP 4: Cleaning Boltz Output Directory ---")
    
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    cleanup_boltz_output(config)
    
    logging.info("--- STEP 4: Finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An unexpected error occurred in Step 4: {e}", exc_info=True)
        exit(1)