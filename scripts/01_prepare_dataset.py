# scripts/01_prepare_dataset.py

import yaml
import logging
from ddg_predictor.data_processing import loaders

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Runs the data preparation pipeline: loading, processing, and saving the dataset.
    """
    logging.info("--- STEP 1: Preparing Dataset ---")

    config = loaders.load_config("params.yaml")
    
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)['data_processing']

    # Call the main logic function from the package
    loaders.load_prepare_save(config)
    
    logging.info("--- STEP 1: Finished ---")

if __name__ == "__main__":
    main()