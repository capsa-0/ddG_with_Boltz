"""
Module: extract_features
Description: Main entry point for feature extraction from Boltz predictions.
Orchestrates Boltz model execution on prepared query files.
"""

import logging
import argparse
from ddg.config.config_loader import ProjectConfig
from ddg.feature_extraction.extraction.run_boltz import run_boltz_predictions

# ----- Setup logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(experiment_config_path: str, names_config_path: str = "ddg/config/internal_config.yaml"):
    """
    Execute Boltz predictions on prepared query files.
    
    Args:
        experiment_config_path: Path to experiment YAML configuration
        names_config_path: Path to internal naming configuration YAML
    """
    
    # ----- PHASE A: CONFIGURATION AND STRUCTURAL PARSING -----
    logger.info(f"Loading configuration from: {experiment_config_path}")
    config = ProjectConfig(
        experiment_yaml_path=experiment_config_path, 
        internal_yaml_path=names_config_path
    )

    logger.info("Running Boltz predictions...")
    run_boltz_predictions(config)

    logger.info(f"Feature extraction complete! Results ready at: {config.raw_features_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using Boltz predictions")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--names-config",
        default="ddg/config/internal_config.yaml",
        help="Path to internal configuration YAML file"
    )
    
    args = parser.parse_args()
    main(args.config, args.names_config)