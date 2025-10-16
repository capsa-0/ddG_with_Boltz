# src/ddg_predictor/feature_extraction/output_cleaner.py

import os
import shutil
import logging
import glob
from tqdm import tqdm

def cleanup_boltz_output(config: dict):
    """
    Reorganizes the Boltz output directory and archives run files.
    It intelligently handles both 'batch' and 'one-by-one' processing modes.
    """
    try:
        fe_config = config['feature_extraction']
        dp_config = config['data_processing']
        
        base_out_dir = fe_config['boltz_flags']['out_dir']

        # Determine the execution mode from the config file
        run_one_by_one = fe_config.get('process_one_by_one', True)

        if run_one_by_one:
            # --- Logic for one-by-one mode ---
            logging.info("Cleaning up in 'one-by-one' mode.")
            # Find all individual boltz_results_* folders
            result_folders = glob.glob(os.path.join(base_out_dir, "boltz_results_*"))
            if not result_folders:
                logging.warning("No 'boltz_results_*' folders found to clean up.")
            
            for folder_path in tqdm(result_folders, desc="Cleaning individual outputs"):
                source_predictions_dir = os.path.join(folder_path, "predictions")
                if not os.path.isdir(source_predictions_dir):
                    continue

                # Move the single prediction folder inside 'predictions'
                for dirname in os.listdir(source_predictions_dir):
                    source_path = os.path.join(source_predictions_dir, dirname)
                    destination_path = os.path.join(base_out_dir, dirname)
                    if os.path.exists(destination_path):
                        shutil.rmtree(destination_path)
                    shutil.move(source_path, destination_path)
                
                # Remove the now-empty intermediate folder
                shutil.rmtree(folder_path)

        else:
            # --- Logic for batch mode (the original logic) ---
            logging.info("Cleaning up in 'batch' mode.")
            queries_dir_basename = os.path.basename(dp_config['queries_dir'])
            boltz_results_dir = os.path.join(base_out_dir, f"boltz_results_{queries_dir_basename}")
            source_predictions_dir = os.path.join(boltz_results_dir, "predictions")

            if not os.path.isdir(source_predictions_dir):
                logging.warning(f"Predictions directory not found at '{source_predictions_dir}'. Skipping cleanup.")
            else:
                subdirs = [d for d in os.listdir(source_predictions_dir) if os.path.isdir(os.path.join(source_predictions_dir, d))]
                logging.info(f"Found {len(subdirs)} prediction directories to move.")
                for dirname in tqdm(subdirs, desc="Organizing results"):
                    source_path = os.path.join(source_predictions_dir, dirname)
                    destination_path = os.path.join(base_out_dir, dirname)
                    if os.path.exists(destination_path):
                        shutil.rmtree(destination_path)
                    shutil.move(source_path, destination_path)

                logging.info(f"Removing intermediate directory: {boltz_results_dir}")
                shutil.rmtree(boltz_results_dir)

        # --- Archive Run Files (runs for both modes) ---
        logging.info(f"Archiving run files to: {base_out_dir}")
        os.makedirs(base_out_dir, exist_ok=True)

        config_source_path = os.path.join(dp_config.get['output_dir'], "params.yaml")
        
        mutations_csv_source_path = os.path.join(dp_config['output_dir'], dp_config['mutations_csv_filename'])
        
        shutil.copy(config_source_path, os.path.join(base_out_dir, "params.yaml"))
        if os.path.exists(mutations_csv_source_path):
            shutil.copy(mutations_csv_source_path, os.path.join(base_out_dir, dp_config['mutations_csv_filename']))
        
        logging.info(f"Cleanup and archiving complete. Final results are in: {base_out_dir}")

    except KeyError as e:
        logging.error(f"Configuration key {e} not found in 'params.yaml'.")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred during cleanup: {e}")
        raise
