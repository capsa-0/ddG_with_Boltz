# scripts/05_process_features.py

import yaml
import logging
from ddg_predictor.feature_processing.feature_differ import EmbeddingDiffer
from ddg_predictor.feature_processing.feature_analyzer import FeatureAnalyzer

from ddg_predictor.data_processing.loaders import load_config
load_config('params.yaml')

# Setup basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    logging.info("--- STEP 5: Processing and Summarizing Features ---")
    
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)

    differ = EmbeddingDiffer(config)
    differ.generate_diffs()
    
    analyzer = FeatureAnalyzer(config)
    df_summary = analyzer.summarize_features()
    analyzer.plot_feature_correlations(df_summary)
    
    logging.info("--- STEP 5: Finished ---")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred during Step 5: {e}", exc_info=True)
        exit(1)