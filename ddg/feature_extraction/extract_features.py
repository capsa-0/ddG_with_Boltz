"""
Module: extract_features
Description: Main entry point for feature extraction from structure-model
predictions. Dispatches the prepared query files to the backbone selected by
`feature_extraction.backbone` in the experiment YAML (default: boltz2).
"""

import logging
import argparse
from ddg.config.config_loader import ProjectConfig
from ddg.feature_extraction.extraction.backbones import get_runner

# ----- Setup logging -----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main(experiment_config_path: str, names_config_path: str = "ddg/config/internal_config.yaml", shard=None):
    """
    Execute the configured backbone's predictions on prepared query files.

    Args:
        experiment_config_path: Path to experiment YAML configuration
        names_config_path: Path to internal naming configuration YAML
        shard: optional (i, n) to process only one shard of the queries
    """

    # ----- PHASE A: CONFIGURATION AND STRUCTURAL PARSING -----
    logger.info(f"Loading configuration from: {experiment_config_path}")
    config = ProjectConfig(
        experiment_yaml_path=experiment_config_path,
        internal_yaml_path=names_config_path
    )

    logger.info("Running %s predictions...", config.backbone)
    get_runner(config.backbone)(config, shard=shard)

    logger.info(f"Feature extraction complete! Results ready at: {config.raw_features_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract embeddings using the configured backbone")
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