# scripts/03_extract_features.py

import yaml
from types import SimpleNamespace
import os
import logging
import glob
from tqdm import tqdm

from ddg_predictor.feature_extraction.boltz_wrapper import run_boltz_prediction

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Runs the feature extraction step using Boltz, either in batch mode or one-by-one.
    """
    logging.info("--- STEP 3: Extracting Features with Boltz ---")
    
    with open("config/params.yaml", "r") as f:
        config_dict = yaml.safe_load(f)
    
    fe_config = config_dict['feature_extraction']
    dp_config = config_dict['data_processing']
    
    config_namespace = SimpleNamespace(**fe_config)
    queries_dir = dp_config['queries_dir']

    # --- Conditional logic based on the new option ---
    
    # .get() is a safe way to read the key, if it doesn't exist, use 'True' by default.
    run_one_by_one = fe_config.get('process_one_by_one', True)

    if run_one_by_one:
        logging.info("Execution mode: One by one.")
        query_files = sorted(glob.glob(os.path.join(queries_dir, "*.yaml")))

        if not query_files:
            raise FileNotFoundError(f"No .yaml query files found in '{queries_dir}'. Please run STEP 2 first.")
        
        logging.info(f"Found {len(query_files)} queries to process.")
        failed_queries = []
        for query_file_path in tqdm(query_files, desc="Processing Boltz queries"):
            try:
                run_boltz_prediction(queries_dir=query_file_path, config=config_namespace)
            except Exception as e:
                logging.error(f"Failed to process query: {os.path.basename(query_file_path)}. Error: {e}")
                failed_queries.append(os.path.basename(query_file_path))
        
        if failed_queries:
            logging.warning(f"Completed with {len(failed_queries)} failed queries: {failed_queries}")

    else:
        logging.info("Execution mode: Batch.")
        if not os.path.isdir(queries_dir) or not os.listdir(queries_dir):
            raise FileNotFoundError(f"Queries directory is empty or not found at '{queries_dir}'. Please run STEP 2 first.")
        
        # Call the function once with the full directory
        run_boltz_prediction(queries_dir=queries_dir, config=config_namespace)

    logging.info("--- STEP 3: Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"A critical error occurred in Step 3: {e}")
        exit(1)