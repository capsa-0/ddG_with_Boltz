# scripts/06_analyze_features.py

import yaml
import os
import logging
from ddg_predictor.visualization.plots import plot_feature_correlations

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Runs the analysis step: generates plots from the feature summary file.
    """
    logging.info("--- STEP 6: Analyzing Features ---")
    
    with open("config/params.yaml", "r") as f:
        config = yaml.safe_load(f)
        
    fe_config = config['feature_extraction']

    # Path to the input CSV file
    summary_csv_path = os.path.join(
        fe_config['boltz_flags']['out_dir'],
        'features_summary.csv')
    
    # Directory where the plot will be saved
    output_plot_dir = fe_config['boltz_flags']['out_dir']

    plot_feature_correlations(
        summary_csv_path=summary_csv_path,
        output_dir=output_plot_dir
    )
    
    logging.info("--- STEP 6: Finished ---")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logging.error(f"An error occurred during Step 6: {e}", exc_info=True)
        exit(1)