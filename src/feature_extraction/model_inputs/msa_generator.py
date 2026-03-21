"""
Module: MsaGenerator
Description: Generates multiple sequence alignments using MMseqs2 server.
"""

import os
import re
import logging
from Bio import SeqIO
from tqdm import tqdm

from external.mmseqs import _run_mmseqs2

logger = logging.getLogger(__name__)


class MsaGenerator:
    """Generates MSAs for protein sequences using MMseqs2."""

    def __init__(self, config, **mmseqs2_kwargs):
        """
        Initialize MSA generator.
        
        Args:
            config: ProjectConfig instance
            **mmseqs2_kwargs: Additional parameters for MMseqs2
        """
        self.config = config
        self.output_dir = config.msa_dir
        self.mmseqs2_kwargs = mmseqs2_kwargs
        os.makedirs(self.output_dir, exist_ok=True)
        logger.debug(f"MSA output directory: {self.output_dir}")

    def get_sequences_from_fasta(self, fasta_path: str) -> dict[str, str]:
        """
        Parse FASTA file into dictionary.
        
        Args:
            fasta_path: Path to FASTA file
            
        Returns:
            Dictionary mapping sequence ID to sequence string
        """
        sequences = {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}
        logger.debug(f"Loaded {len(sequences)} sequences from {fasta_path}")
        return sequences

    @staticmethod
    def _safe_prefix(seq_id: str) -> str:
        """
        Sanitize sequence ID for use as file prefix.
        
        Args:
            seq_id: Original sequence ID
            
        Returns:
            Safe prefix for temporary files
        """
        sanitized = re.sub(r"[^A-Za-z0-9_.-]", "_", seq_id)
        return f"tmp_{sanitized}"

    def generate_for_sequence(self, seq_id: str, sequence: str):
        """
        Generate MSA for single sequence.
        
        Args:
            seq_id: Sequence identifier
            sequence: Protein sequence string
        """
        output_path = os.path.join(self.output_dir, f"{seq_id}.a3m")
        if os.path.exists(output_path):
            logger.debug(f"MSA for {seq_id} already exists. Skipping.")
            return

        try:
            a3m_lines = _run_mmseqs2(
                x=sequence,
                prefix=self._safe_prefix(seq_id),
                **self.mmseqs2_kwargs,
            )
            msa_content = a3m_lines[0]

            lines = msa_content.split('\n')
            if lines and lines[0].startswith('>'):
                lines[0] = f'>{seq_id}'

            with open(output_path, 'w') as f:
                f.write('\n'.join(lines))
            logger.debug(f"Generated MSA for {seq_id}")
        except Exception as e:
            logger.error(f"Failed to generate MSA for {seq_id}: {e}")

    def generate_msas_for_multifasta(self):
        """Generate MSAs for all sequences in multifasta file."""
        sequences = self.get_sequences_from_fasta(self.config.multifasta_path)
        logger.info(f"Starting MSA generation for {len(sequences)} sequences")

        for seq_id, sequence in tqdm(sequences.items(), desc="Generating MSAs"):
            self.generate_for_sequence(seq_id, sequence)