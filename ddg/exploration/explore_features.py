"""
Module: explore_features
Description: Main entry point for exploration analysis pipeline.
Orchestrates feature analysis and visualization generation.
"""

import logging
import argparse
from ddg.config.config_loader import ProjectConfig
from ddg.datasets.boltz_dataset import BoltzDataset
from ddg.exploration.feature_analysis.feature_analyzer import FeatureAnalyzer

# ----- Setup logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(experiment_config_path: str, names_config_path: str = "ddg/config/internal_config.yaml"):
    """
    Execute feature analysis and visualization pipeline.
    
    Args:
        experiment_config_path: Path to experiment YAML configuration
        names_config_path: Path to internal naming configuration YAML
    """
    logger.info(f"Loading configuration from: {experiment_config_path}")
    config = ProjectConfig(
        experiment_yaml_path=experiment_config_path, 
        internal_yaml_path=names_config_path
    )

    logger.info("Starting feature analysis")
    analyzer = FeatureAnalyzer(config)
    analyzer.analyze()
    logger.info("Feature analysis complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute feature exploration and analysis pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration")

    args = parser.parse_args()
    main(args.config)