"""
Module: MSAModifier
Description: Modifies multiple sequence alignment files (trimming, mutations, etc).
"""

import logging
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import pandas as pd

logger = logging.getLogger(__name__)


class MsaModifier:
    """Modifies MSA records (trimming, mutations, sequence flattening)."""

    def __init__(self, msa_path):
        """
        Initialize modifier with MSA file.
        
        Args:
            msa_path: Path to A3M/FASTA format MSA file
        """
        self.msa_path = msa_path
        self.records = list(SeqIO.parse(msa_path, "fasta"))
        logger.debug(f"Loaded {len(self.records)} sequences from {msa_path}")

    @classmethod
    def from_cache(cls, cached):
        """Build a modifier from cached (id, description, sequence) triples.

        Generating one mutant MSA per mutation otherwise re-parses the whole base
        alignment every time — for a 400 aa protein with 1000 MSA rows and 7562
        mutations that is ~45 min of pure re-parsing. The cache is built once per
        base MSA and this rebuilds cheap records from it, so mutate/save logic
        stays in one place.
        """
        obj = cls.__new__(cls)
        obj.msa_path = None
        obj.records = [SeqRecord(Seq(seq), id=rid, description=desc)
                       for rid, desc, seq in cached]
        return obj

    def as_cache(self):
        """(id, description, sequence) triples for from_cache()."""
        return [(r.id, r.description, str(r.seq)) for r in self.records]

    def keep_first_n_sequences(self, n):
        """
        Keep only first n sequences in alignment.
        
        Args:
            n: Maximum number of sequences to keep
        """
        original_count = len(self.records)
        self.records = self.records[:n]
        logger.debug(f"Trimmed sequences from {original_count} to {len(self.records)}")

    def mutate_position(self, old_aa, position, new_aa, only_first_row=False):
        """
        Introduce mutation at specific alignment position.
        
        Args:
            old_aa: Expected amino acid at position
            position: Zero-indexed column position in alignment
            new_aa: Replacement amino acid
            only_first_row: If True, only mutate first sequence
        """
        for i, record in enumerate(self.records):
            if only_first_row and i > 0:
                break
            
            seq_list = list(str(record.seq))
            
            if position < len(seq_list):
                if seq_list[position] != '-':
                    seq_list[position] = new_aa
                record.seq = Seq("".join(seq_list))

        # ----- Update first sequence name with mutation -----
        new_name = self.records[0].id + f"_{old_aa}{position+1}{new_aa}"
        self.records[0].id = new_name
        self.records[0].description = ""
        logger.debug(f"Applied mutation at position {position}: {old_aa}{position+1}{new_aa}")

    def flatten_sequences(self):
        """Remove line breaks from sequences (single-line format)."""
        for record in self.records:
            record.seq = Seq(str(record.seq).replace("\n", ""))
        logger.debug("Flattened sequences to single lines")

    def save(self, output_path):
        """
        Save modified MSA to file.
        
        Args:
            output_path: Output file path
        """
        with open(output_path, "w") as f:
            for record in self.records:
                f.write(f">{record.id}")
                if record.description and record.description != record.id:
                    f.write(f" {record.description}")
                f.write("\n")
                f.write(str(record.seq) + "\n")
        logger.debug(f"Saved {len(self.records)} sequences to {output_path}")


class MSADirectoryModifier:
    """Batch modify all MSA files in a directory."""

    def __init__(self, config):
        """
        Initialize directory modifier.
        
        Args:
            config: ProjectConfig instance
        """
        self.config = config
        self.msa_dir = config.msa_dir
        self.mutations_df = pd.read_csv(config.mutations_df_path)
        self.only_first_row = config.msa_mutation_strategy == "mutate_first_row"
        logger.debug(f"Initialized MSA directory modifier for {self.msa_dir}")

    def apply_trimming_to_directory(self):
        """Trim all MSA files in directory to max_sequences threshold."""
        max_sequences = self.config.max_msa_sequences
        logger.info(f"Trimming all MSAs to {max_sequences} sequences maximum")
        
        for msa_file in self.msa_dir.glob("*.a3m"):
            modifier = MsaModifier(msa_file)
            modifier.keep_first_n_sequences(max_sequences)
            modifier.save(msa_file)

    def get_mutations(self) -> dict:
        """
        Extract mutations grouped by wild-type ID.
        
        Returns:
            Dictionary mapping wt_id to list of mutations
        """
        mutations = {}
        for _, row in self.mutations_df.iterrows():
            wt_id = row['wt_id']
            mutation = row['mutation']
            if wt_id not in mutations:
                mutations[wt_id] = []
            mutations[wt_id].append(mutation)
        return mutations
    
    def apply_mutations_to_directory(self):
        """
        Apply each mutation to wild-type MSA and save separately.
        Each file becomes: wt_id_mutation.a3m

        Existing mutant MSAs are left alone, so re-running prepare on an experiment
        that already has them is fast. Without this a resumed run rebuilds every
        mutant alignment from scratch (~1.8 h and ~3.4 GB rewritten for a 7562-mutation
        scan) purely to reproduce files that are already on disk.
        """
        mutations = self.get_mutations()
        logger.info("Applying mutations to MSA files")
        written = skipped = 0

        # Materialise the listing first: this loop writes .a3m files into the very
        # directory it is iterating.
        for msa_file in sorted(self.msa_dir.glob("*.a3m")):
            wt_id = msa_file.stem
            if wt_id not in mutations:
                continue
            cache = None
            for mutation in mutations[wt_id]:
                new_path = msa_file.parent / f"{wt_id}_{mutation}.a3m"
                if new_path.exists() and new_path.stat().st_size > 0:
                    skipped += 1
                    continue
                if cache is None:                     # parse the base MSA once
                    cache = MsaModifier(msa_file).as_cache()
                modifier = MsaModifier.from_cache(cache)

                # Parse mutation: e.g., "Y1A" -> position=0, old_aa="Y", new_aa="A"
                position = int(mutation[1:-1]) - 1
                old_aa = mutation[0]
                new_aa = mutation[-1]

                modifier.mutate_position(old_aa, position, new_aa,
                                         only_first_row=self.only_first_row)
                modifier.save(new_path)
                written += 1

        logger.info("MSA mutants: %d written, %d already present (skipped)",
                    written, skipped)

    def flatten_sequences(self):
        """Flatten all sequences in all MSA files to single lines."""
        logger.info("Flattening all sequences in MSA directory")
        for msa_file in self.msa_dir.glob("*.a3m"):
            modifier = MsaModifier(msa_file)
            modifier.flatten_sequences()
            modifier.save(msa_file)