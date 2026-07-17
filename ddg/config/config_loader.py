"""
Module: ProjectConfig
Description: Loads and manages experiment configuration from YAML files.
Merges experiment-specific parameters with internal naming conventions.
"""

import yaml
import logging
import shutil
from pathlib import Path
import os

logger = logging.getLogger(__name__)


class ProjectConfig:
    def __init__(self, experiment_yaml_path: str, internal_yaml_path: str = "configs/names_config.yaml"):
        """
        Initialize ProjectConfig by loading and merging YAML configuration files.
        
        Args:
            experiment_yaml_path: Path to experiment-specific configuration YAML
            internal_yaml_path: Path to internal naming conventions YAML
        """
        # ----- Load internal naming configuration -----
        with open(internal_yaml_path, "r") as f:
            names_config = yaml.safe_load(f)
            self._dirs = names_config.get('directories', {})
            self._files = names_config.get('filenames', {})

        # ----- Load experiment-specific configuration -----
        with open(experiment_yaml_path, "r") as f:
            self.exp_config = yaml.safe_load(f)

        # ----- Map frequent-use variables -----
        self.mode = self.exp_config['head']['mode']
        self.experiment_name = self.exp_config['head']['experiment_name']
        
        data_proc = self.exp_config['data_processing']
        self.overwrite = data_proc['overwrite']
        self.raw_data_path = Path(data_proc['raw_data_path'])
        self.dataset_type = data_proc['dataset_type']
        self.msa_strategy = data_proc['msa_strategy']
        self.msa_mutation_strategy = data_proc['msa_mutation_strategy']
        self.max_msa_sequences = data_proc['max_msa_sequences']
        # Single-sequence mode: skip MSA generation and tell Boltz to run
        # without an MSA (`msa: empty`). Default False -> normal MSA pipeline.
        self.no_msa = bool(data_proc.get('no_msa', False))
                
        self.process_one_by_one = self.exp_config['feature_extraction']['process_one_by_one']
        self.boltz_flags = self.exp_config['feature_extraction']['boltz_flags']
        self.training_params = self.exp_config.get('training', {})

    # ----- BASE PATHS (Dynamic based on names_config) -----
    
    @property
    def exp_processed_dir(self) -> Path:
        """Return experiment-specific processed data directory."""
        return Path(self._dirs['processed_dir']) / self.experiment_name

    # ----- SUB-DIRECTORIES (Inside processed_dir) -----

    @property
    def msa_dir(self) -> Path:
        """Return directory for multiple sequence alignment files."""
        return self.exp_processed_dir / self._dirs['msa_output_dir']

    @property
    def queries_dir(self) -> Path:
        """Return directory for Boltz query YAML files."""
        return self.exp_processed_dir / self._dirs['queries_dir']
    
    @property
    def raw_features_dir(self) -> Path:
        """Return directory for raw feature extraction output."""
        return self.exp_processed_dir / self._dirs['raw_features_dir']

    # ----- FILES (Inside processed_dir) -----

    @property
    def mutations_df_path(self) -> Path:
        """Return path to mutations dataframe CSV file."""
        return self.exp_processed_dir / self._files['mutations_df']

    @property
    def metadata_df_path(self) -> Path:
        """Return path to metadata dataframe CSV file."""
        return self.exp_processed_dir / self._files['metadata_df']
        
    @property
    def multifasta_path(self) -> Path:
        """Return path to multifasta FASTA file."""
        return self.exp_processed_dir / self._files['multifasta']

    # ----- RELATIVE PATH CONFIGURATION (For Boltz YAML) -----

    @property
    def msa_dir_relative(self) -> str:
        """Return MSA directory as relative path from project root."""
        try:
            return str(self.msa_dir.relative_to(Path.cwd()))
        except ValueError:
            logger.debug(f"MSA directory outside cwd, using absolute path")
            return str(self.msa_dir)

    @property
    def queries_dir_relative(self) -> str:
        """Return queries directory as relative path from project root."""
        try:
            return str(self.queries_dir.relative_to(Path.cwd()))
        except ValueError:
            logger.debug(f"Queries directory outside cwd, using absolute path")
            return str(self.queries_dir)

    def prepare_processed_directory(self):
        """
        Prepare the processed data directory for experiment.
        Removes existing directory if overwrite flag is set.
        
        Raises:
            FileNotFoundError: If raw data file does not exist
        """
        if not self.raw_data_path.exists():
            raise FileNotFoundError(f"Raw data file not found at {self.raw_data_path}")
        
        processed_dir = self.exp_processed_dir
        if processed_dir.exists():
            if self.overwrite:
                logger.warning(f"Overwrite flag set: Deleting existing directory {processed_dir}")
                shutil.rmtree(processed_dir)
        
        os.makedirs(processed_dir, exist_ok=True)
        logger.debug(f"Processed directory ready: {processed_dir}")