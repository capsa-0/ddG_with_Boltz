import os
import re
from pathlib import Path
import pandas as pd
from abc import ABC, abstractmethod


class MsaMutator(ABC):
    """
    Abstract base class for MSA mutation.
    Subclasses must implement read_msa and save_msa for specific file formats.
    """

    def __init__(self, msa_dir: str, mutations_df: pd.DataFrame):
        """
        Args:
            msa_dir (str): Directory containing MSA files.
            mutations_df (pd.DataFrame): DataFrame with columns ['sequence_id', 'mutation', 'ddg']
        """
        self.msa_dir = msa_dir
        # Group mutations by sequence_id
        self.mutations_by_id = mutations_df.groupby("sequence_id")["mutation"].apply(list).to_dict()


    def mutate_directory(self):
        """
        Iterate over all MSA files in the directory and apply mutations
        that correspond to the sequence ID (file stem).
        """

        for msa_filename in os.listdir(self.msa_dir):

            msa_path = os.path.join(self.msa_dir, msa_filename)
            seq_id = Path(msa_filename).stem

            if seq_id not in self.mutations_by_id:
                continue  # Skip files with no listed mutations

            msa_records = self.read_msa(msa_path)

            for mutation in self.mutations_by_id[seq_id]:
                mutation_dict = {seq_id: mutation}
                mutated_records = self.apply_mutation(msa_records, mutation_dict)
                output_path = os.path.join(self.msa_dir, f"{seq_id}_{mutation}{self.file_extension()}")

                self.save_msa(mutated_records, output_path)


    @abstractmethod
    def read_msa(self, path: str):
        """Read an MSA file and return list of (header, sequence) tuples."""
        pass


    @abstractmethod
    def save_msa(self, records, path: str):
        """Save list of (header, sequence) tuples to an MSA file."""
        pass

    @abstractmethod
    def file_extension(self):
        """Return the expected file extension for this MSA type (e.g., '.a3m')."""
        pass


    @staticmethod
    def parse_mutation(mutation: str):
        """Parse a mutation string like 'A23T' into (orig_residue, pos, new_residue)."""
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation)
        if not match:
            raise ValueError(f"Invalid mutation format: {mutation}")
        orig_res, pos, new_res = match.groups()
        return orig_res, int(pos), new_res

    @staticmethod
    def ungapped_sequence(seq_with_gaps: str):
        """Return sequence without gaps, keeping only uppercase letters (query)."""
        return "".join([aa for aa in seq_with_gaps if aa.isupper()])

    @staticmethod
    def map_query_position_to_alignment(seq_with_gaps: str, query_pos: int):
        """Map ungapped query position to aligned sequence position (1-based)."""
        count = 0
        for i, aa in enumerate(seq_with_gaps, start=1):
            if aa.isupper():
                count += 1
                if count == query_pos:
                    return i
        raise ValueError(f"Query position {query_pos} exceeds sequence length.")

    def apply_mutation(self, records, mutation_dict: dict):
        """
        Apply mutation(s) to MSA records, given a dict {sequence_id: mutation}.
        Supports only one mutation per call.
        """
        pass



class A3mMutator(MsaMutator):
    """
    Concrete implementation for A3M MSA files.
    """

    def read_msa(self, path: str):
        records = []
        header = None
        seq_parts = []

        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is not None:
                        records.append((header, "".join(seq_parts)))
                    header = line
                    seq_parts = []
                else:
                    seq_parts.append(line.strip())
            if header is not None:
                records.append((header, "".join(seq_parts)))
        return records

    def save_msa(self, records, path: str):
        with open(path, "w") as f:
            for header, seq in records:
                f.write(f"{header}\n{seq}\n")

    def file_extension(self):
        return ".a3m"
    
    def apply_mutation(self, records, mutation_dict: dict):
        """
        Apply mutation(s) to MSA records, given a dict {sequence_id: mutation}.
        Supports only one mutation per call.
        """
        sequence_id, mutation = list(mutation_dict.items())[0]
        orig_res, query_pos, new_res = self.parse_mutation(mutation)

        query_seq = records[0][1]
        ungapped_query = self.ungapped_sequence(query_seq)

        if ungapped_query[query_pos - 1].upper() != orig_res:
            raise ValueError(
                f"Original residue at position {query_pos} does not match {orig_res}."
            )

        aligned_pos = self.map_query_position_to_alignment(query_seq, query_pos)

        mutated_records = []
        uniprot_id = Path(records[0][0].split()[0]).stem
        new_header = f"{uniprot_id}_{mutation}"

        for i, (header, seq) in enumerate(records):
            if aligned_pos - 1 < len(seq):
                aa_at_pos = seq[aligned_pos - 1]
                if aa_at_pos != "-":
                    seq = (
                        seq[:aligned_pos - 1]
                        + (new_res.lower() if aa_at_pos.islower() else new_res.upper())
                        + seq[aligned_pos:]
                    )
            mutated_records.append((new_header, seq) if i == 0 else (header, seq))

        return mutated_records
