"""
Module: feature_analyzer
Description: Analyzes Boltz embeddings and generates feature extraction visualizations.
"""

import os
import re
import logging
import pandas as pd
from tqdm import tqdm

from .extractors import extract_features
from .plots import plot_umap, plot_correlations, plot_correlation_summary
from ddg.datasets.boltz_dataset import BoltzDataset

logger = logging.getLogger(__name__)


def make_feature_dataset(config):
    """
    Choose the embedding source for feature extraction.

    config feature.source: 'raw' | 'slim' | 'auto' (default 'auto').
    'auto' uses the slim store when it exists, else the raw Boltz output.
    """
    source = config.exp_config.get("feature", {}).get("source", "auto")
    slim_exists = (config.exp_processed_dir / "slim").exists()
    if source == "slim" or (source == "auto" and slim_exists):
        from ddg.storage.slim_store import SlimBoltzDataset
        logger.info("Feature source: slim store")
        return SlimBoltzDataset(config)
    logger.info("Feature source: raw Boltz output")
    return BoltzDataset(config)


class FeatureAnalyzer:
    """Analyzes Boltz embeddings and extracts mutation-related features."""

    def __init__(self, config):
        """
        Initialize feature analyzer.
        
        Args:
            config: ProjectConfig instance
        """
        logger.debug("Initializing FeatureAnalyzer")
        self.config = config
        self.dataset = make_feature_dataset(config)
        
        # ----- Output configuration -----
        self.output_dir = config.exp_processed_dir
        self.parquet_path = os.path.join(self.output_dir, "features_summary.parquet")
        self.plots_dir = os.path.join(self.output_dir, "exploration_plots")
        
        logger.debug(f"Output directory: {self.output_dir}")
        logger.debug(f"Parquet file path: {self.parquet_path}")

    @staticmethod
    def _parse_mutation_pos(mutation_str: str) -> int:
        """
        Extract zero-indexed position from mutation string.
        
        Args:
            mutation_str: Mutation string like "P8A"
            
        Returns:
            Zero-indexed position
            
        Raises:
            ValueError: If mutation string format invalid
        """
        match = re.search(r'(\d+)', mutation_str)
        if match:
            idx = int(match.group(1)) - 1
            logger.debug(f"Parsed mutation '{mutation_str}' to position {idx}")
            return idx
        
        logger.error(f"Regex parse failed for mutation: {mutation_str}")
        raise ValueError(f"Could not extract position from: {mutation_str}")

    def analyze(self):
        """
        Orchestrate complete feature analysis workflow.
        Loads or computes features, generates visualizations.
        """
        logger.info(f"Starting feature extraction for {len(self.dataset)} samples")
        
        # ----- Load existing features or compute new -----
        if os.path.exists(self.parquet_path):
            logger.info(f"Found existing features. Loading from {self.parquet_path}")
            df_features = pd.read_parquet(self.parquet_path)
            logger.debug(f"Loaded features with shape: {df_features.shape}")
        else:
            logger.info("No existing features found. Computing from scratch")
            df_features = self._gather_and_save_features()

        # ----- Generate visualizations -----
        logger.info("Generating correlation summary plot")
        plot_correlation_summary(df_features, output_dir=self.plots_dir)

        logger.info("Generating UMAP projection")
        plot_umap(df_features, output_dir=self.plots_dir)
        
        logger.info("Generating correlation scatter plots")
        plot_correlations(df_features, output_dir=os.path.join(self.plots_dir, "scatter_plots"))
        
        logger.info("Analysis complete!")

    def _gather_and_save_features(self) -> pd.DataFrame:
        """
        Extract features from all samples and save to parquet.
        
        Returns:
            DataFrame with all extracted features
        """
        data_records = []
        
        for i in tqdm(range(len(self.dataset)), desc="Processing embeddings"):
            logger.debug(f"Processing sample {i}")
            
            # ----- Load tensors from dataset -----
            try:
                sample = self.dataset[i]
                logger.debug(f"Loaded tensors for sample {i}")
            except Exception as e:
                logger.error(f"Error loading tensors at index {i}: {e}")
                continue
                
            # ----- Extract metadata -----
            wt_id = sample['wt_id']
            mut_id = sample['mut_id']
            mut_str = sample['mutation']
            ddg = sample['ddg'].item()
            
            logger.debug(f"Metadata: mut_id={mut_id}, wt_id={wt_id}, mutation={mut_str}, ddg={ddg}")
            
            try:
                mut_pos = self._parse_mutation_pos(mut_str)
            except ValueError:
                logger.warning(f"Skipping {mut_id} due to parse error: {mut_str}")
                continue

            # ----- Compute mathematical features -----
            logger.debug(f"Computing features for {mut_id} at position {mut_pos}")
            features_dict = extract_features(sample, mut_pos, window_size=5)
            logger.debug(f"Extracted {len(features_dict)} features")
            
            # ----- Inject metadata into feature dictionary -----
            features_dict["mut_id"] = mut_id
            features_dict["wt_id"] = wt_id
            features_dict["mutation"] = mut_str
            features_dict["ddg"] = ddg
            
            data_records.append(features_dict)
            logger.debug(f"Completed processing for {mut_id}")
            
        # ----- Convert to DataFrame -----
        logger.info(f"Converting {len(data_records)} records to DataFrame")
        df = pd.DataFrame(data_records)
        
        # ----- Reorder columns for readability -----
        metadata_cols = ["mut_id", "wt_id", "mutation", "ddg"]
        feature_cols = [c for c in df.columns if c not in metadata_cols]
        df = df[metadata_cols + feature_cols]
        
        logger.debug(f"DataFrame shape: {df.shape}")

        # ----- Save to parquet -----
        logger.info(f"Creating output directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        df.to_parquet(self.parquet_path, index=False)
        logger.info(f"Features saved to: {self.parquet_path}")
        
        return df