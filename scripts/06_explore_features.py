# scripts/06_explore_features.py

import yaml
import logging
import argparse
import os
from ddg_predictor.exploration.data_explorer import ResultAnalyzer

from ddg_predictor.data_processing.loaders import load_config
load_config('params.yaml')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main(args):
    """
    Runs the post-pipeline analysis: PCA and feature correlation plots.
    """
    logging.info("--- STEP 6: Analyzing Results ---")

    # Verificar que el directorio exista
    if not os.path.isdir(args.directory):
        logging.error(f"Provided directory not found: {args.directory}")
        exit(1)

    # Ruta al archivo params.yaml dentro del directorio
    params_path = os.path.join(args.directory, "params.yaml")

    if not os.path.isfile(params_path):
        logging.error(f"params.yaml not found in {args.directory}")
        exit(1)

    # Cargar configuración
    with open(params_path, "r") as f:
        config = yaml.safe_load(f)

    # Inicializar el analizador
    analyzer = ResultAnalyzer(config)

    # Ejecutar los análisis seleccionados
    if args.run_pca:
        analyzer.run_global_pca_analysis()

    if args.run_correlations:
        analyzer.run_feature_correlation_analysis()

    logging.info("--- STEP 6: Finished ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run analysis plots on pipeline results.")
    parser.add_argument("directory", type=str, help="Path to the directory containing params.yaml")
    parser.add_argument("--run_pca", action="store_true", help="Run the global PCA analysis on raw diff tensors.")
    parser.add_argument("--run_correlations", action="store_true", help="Run the feature correlation analysis on the summary file.")

    # Si no se especifica ningún flag, ejecutar ambos análisis por defecto
    args = parser.parse_args()
    if not args.run_pca and not args.run_correlations:
        args.run_pca = True
        args.run_correlations = True

    try:
        main(args)
    except Exception as e:
        logging.error(f"An error occurred during analysis: {e}", exc_info=True)
        exit(1)
