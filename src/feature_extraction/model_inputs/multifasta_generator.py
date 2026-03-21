"""
Module: MultifastaGenerator
Description: Generates multifasta FASTA files from mutation dataset.
"""

import logging
import pandas as pd

logger = logging.getLogger(__name__)


class MultifastaGenerator:
    """Generate FASTA files containing wild-type sequences."""

    def __init__(self, config):
        """
        Initialize generator with experiment configuration.
        
        Args:
            config: ProjectConfig instance
        """
        self.config = config
        self.mutations_df = pd.read_csv(config.mutations_df_path)
        self.fasta_dict = {}

    def define_samples(self):
        """
        Extract unique wild-type sequences from mutations dataframe.
        Follows configured MSA strategy.
        """
        logger.debug(f"Dataset columns: {list(self.mutations_df.columns)}")
        logger.debug(f"Dataset shape: {self.mutations_df.shape}")
        
        if self.config.msa_strategy == "mutate_wt_msa":
            self.fasta_dict = (
                self.mutations_df[['wt_id', 'sequence_wt']]
                .drop_duplicates()
                .set_index('wt_id')['sequence_wt']
                .to_dict()
            )
            logger.info(f"Extracted {len(self.fasta_dict)} unique wild-type sequences")
        
    def generate_multifasta(self):
        """
        Write FASTA file with all wild-type sequences.
        Outputs to path defined in config.multifasta_path
        """
        if not self.fasta_dict:
            logger.warning("No sequences defined. Call define_samples() first.")
            return
            
        with open(self.config.multifasta_path, 'w') as fasta_file:
            for seq_id, sequence in self.fasta_dict.items():
                fasta_file.write(f">{seq_id}\n{sequence}\n")
        
        logger.info(f"Wrote {len(self.fasta_dict)} sequences to {self.config.multifasta_path}")