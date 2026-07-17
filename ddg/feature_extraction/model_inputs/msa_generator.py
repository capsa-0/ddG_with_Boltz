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

    def _write_a3m(self, seq_id: str, a3m_content: str) -> None:
        """Write one sequence's a3m, renaming the query header to seq_id."""
        lines = a3m_content.split('\n')
        if lines and lines[0].startswith('>'):
            lines[0] = f'>{seq_id}'
        with open(os.path.join(self.output_dir, f"{seq_id}.a3m"), 'w') as f:
            f.write('\n'.join(lines))

    def write_single_sequence_msas(self):
        """Write single-sequence a3m files (no MSA search).

        For `no_msa` runs we skip the MMseqs2 server entirely and emit one
        query-only a3m per WT sequence (`>{seq_id}` + the bare sequence). This
        carries the WT sequence through the same downstream machinery
        (trimming, mutation application, YAML conversion) as a real MSA, so the
        per-mutation query sequences are built identically; only the `msa:`
        field in the final Boltz YAML differs (set to `empty`). Resumable: an
        existing .a3m is left untouched.
        """
        sequences = self.get_sequences_from_fasta(self.config.multifasta_path)
        logger.info(f"no_msa: writing {len(sequences)} single-sequence a3m files "
                    f"(skipping MMseqs2 server)")
        written = 0
        for seq_id, sequence in sequences.items():
            output_path = os.path.join(self.output_dir, f"{seq_id}.a3m")
            if os.path.exists(output_path):
                continue
            with open(output_path, 'w') as f:
                f.write(f">{seq_id}\n{sequence}\n")
            written += 1
        logger.info(f"no_msa: wrote {written} single-sequence a3m files to "
                    f"{self.output_dir}")

    def generate_msas_for_multifasta(self):
        """Generate MSAs for all sequences in the multifasta file.

        Submits sequences to the MMseqs2 server in BATCHES (one server job per
        chunk) instead of one round-trip per sequence. The ColabFold API takes a
        list, dedups it, and returns one a3m per input, so a chunk of C sequences
        costs a single ~150s queue wait rather than C of them -- this is what
        dominated `prepare` at scale. Chunking (vs one giant submission) keeps
        each job within server limits and makes the run resumable per sequence:
        already-written .a3m files are skipped, so a rerun only fetches leftovers.

        Chunk size is `data_processing.msa_batch_size` (default 50).
        """
        sequences = self.get_sequences_from_fasta(self.config.multifasta_path)
        logger.info(f"Starting MSA generation for {len(sequences)} sequences")

        # Resumability: only fetch sequences that don't already have an .a3m.
        pending = {sid: seq for sid, seq in sequences.items()
                   if not os.path.exists(os.path.join(self.output_dir, f"{sid}.a3m"))}
        skipped = len(sequences) - len(pending)
        if skipped:
            logger.info(f"Skipping {skipped} sequences with existing MSAs")
        if not pending:
            logger.info("All MSAs already present; nothing to fetch")
            return

        batch_size = int(self.config.exp_config.get("data_processing", {})
                         .get("msa_batch_size", 50))
        ids = list(pending.keys())
        logger.info(f"Fetching {len(ids)} MSAs in batches of {batch_size}")

        failures = 0
        for start in tqdm(range(0, len(ids), batch_size), desc="MSA batches"):
            chunk_ids = ids[start:start + batch_size]
            chunk_seqs = [pending[sid] for sid in chunk_ids]
            try:
                a3m_lines = _run_mmseqs2(
                    x=chunk_seqs,
                    prefix=f"tmp_msa_batch_{start:06d}",
                    **self.mmseqs2_kwargs,
                )
                for seq_id, a3m in zip(chunk_ids, a3m_lines):
                    self._write_a3m(seq_id, a3m)
            except Exception as e:  # keep earlier chunks; retry leftovers on rerun
                failures += len(chunk_ids)
                logger.error(f"MSA batch starting at {start} failed: {e}")

        if failures:
            raise RuntimeError(
                f"{failures}/{len(ids)} MSAs failed; rerun `prepare` to retry "
                f"(already-written MSAs are skipped)")
        logger.info(f"Wrote {len(ids)} MSAs to {self.output_dir}")