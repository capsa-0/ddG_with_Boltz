"""
Module: generate_queries
Description: Main entry point for data preparation and query generation pipeline.
Orchestrates multifasta generation, MSA generation, mutation application, and YAML conversion.
"""

import logging
import argparse
from ddg.config.config_loader import ProjectConfig
from ddg.datasets.load_input_dataset import load_dataset
from ddg.feature_extraction.model_inputs.multifasta_generator import MultifastaGenerator
from ddg.feature_extraction.model_inputs.msa_generator import MsaGenerator
from ddg.feature_extraction.model_inputs.msa_modifier import MSADirectoryModifier
from ddg.feature_extraction.model_inputs.queries_generator import MsaToBoltzYamlConverter

logger = logging.getLogger(__name__)


def main(experiment_config_path: str, names_config_path: str = "ddg/config/internal_config.yaml"):
    """
    Execute complete data preparation and query generation pipeline.
    
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

    logger.info("Preparing and cleaning directories...")
    config.prepare_processed_directory()

    logger.info(f"Loading dataset type '{config.dataset_type}'...")
    dataset = load_dataset(
        dataset_type=config.dataset_type,
        csv_file=config.raw_data_path,
        mode=config.mode
    )

    logger.info("Processing and saving consolidated DataFrames...")
    mutations_df = dataset.to_mutations_dataframe()
    metadata_df = dataset.to_metadata_dataframe()
    mutations_df.to_csv(config.mutations_df_path, index=False)
    metadata_df.to_csv(config.metadata_df_path, index=False)
    logger.debug(f"Saved {len(mutations_df)} mutations to {config.mutations_df_path}")
    
    # ----- PHASE B: MODEL INPUT GENERATION (BOLTZ) -----
    logger.info("Starting Multifasta WT generation...")
    multifasta_generator = MultifastaGenerator(config)
    multifasta_generator.define_samples() 
    multifasta_generator.generate_multifasta()

    logger.info("Generating base MSAs (this may take some time)...")
    msa_generator = MsaGenerator(config)
    msa_generator.generate_msas_for_multifasta()

    logger.info("Applying mutations and trimming to MSAs...")
    msa_modifier = MSADirectoryModifier(config)
    msa_modifier.apply_trimming_to_directory()
    msa_modifier.apply_mutations_to_directory()
    msa_modifier.flatten_sequences() 

    logger.info("Converting MSAs to YAML files for Boltz...")
    yaml_converter = MsaToBoltzYamlConverter(config)
    yaml_converter.convert_a3m_dir_to_yaml()

    logger.info(f"Data preparation pipeline completed! Ready at: {config.exp_processed_dir}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    parser = argparse.ArgumentParser(description="Execute data preparation and MSA generation pipeline")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML configuration")

    args = parser.parse_args()
    main(args.config)