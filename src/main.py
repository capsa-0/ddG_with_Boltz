"""
Module: extract_features
Description: Main entry point for feature extraction from Boltz predictions.
Orchestrates Boltz model execution on prepared query files.
"""

import logging
import argparse
from src.config.config_loader import ProjectConfig
from src.feature_extraction.generate_queries import main as generate_queries 
from src.feature_extraction.extract_features import main as extract_features

# ----- Setup logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract features using Boltz predictions")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to experiment configuration YAML file"
    )
    parser.add_argument(
        "--names-config",
        default="src/config/internal_config.yaml",
        help="Path to internal configuration YAML file"
    )
    
    args = parser.parse_args()
    generate_queries(args.config, args.names_config)
    extract_features(args.config, args.names_config)