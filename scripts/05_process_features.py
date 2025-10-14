# scripts/05_process_features.py

import yaml
import logging
from ddg_predictor.feature_processing.feature_diff import FeatureDiffer

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("--- STEP 5: Processing and Summarizing Features ---")
    
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    differ = FeatureDiffer(config)
    differ.generate_diffs()
    
    # Choose the features you want to calculate
    # You can try different combinations in each run
    desired_features = ["entropy", "gini", "sum", "mean", "max", "std"]
    
    differ.calculate_deltas_summary(aggregations=desired_features)
    
    logging.info("--- STEP 5: Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred during Step 5: {e}", exc_info=True)
        exit(1)