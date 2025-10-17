# scripts/01_prepare_dataset.py

import yaml
import logging
import os  
from ddg_predictor.data_processing import loaders

from ddg_predictor.data_processing.loaders import load_config
load_config('params.yaml')

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Runs the data preparation pipeline: loading, processing, and saving the dataset.
    First, it checks if the output directory already exists.
    """
    logging.info("--- STEP 1: Preparing Dataset ---")

    # Load the data processing configuration from your yaml file
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)['data_processing']

    output_dir = config['output_dir']

    # --- Check if the output directory already exists ---
    if os.path.exists(output_dir):
        logging.info(f"Output directory '{output_dir}' already exists. Skipping dataset preparation.")
        return  # Exit the function if the directory is found

    # If the directory doesn't exist, proceed with data preparation
    logging.info(f"Output directory not found. Proceeding to create it and process data.")
    
    # Call the main logic function from the package
    loaders.load_prepare_save(config)
    
    logging.info("--- STEP 1: Finished ---")

if __name__ == "__main__":
    main()