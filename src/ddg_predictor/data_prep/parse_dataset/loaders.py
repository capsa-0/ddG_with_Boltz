from abc import ABC, abstractmethod
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
import os
from Bio.SeqRecord import SeqRecord

from sequence_resolver import SequenceResolver


class BaseLoader(ABC):
    """
    Abstract base class for dataset loaders.
    Provides a standard interface for loading, processing, and saving datasets.
    """

    def __init__(self, raw_path: str, output_dir: str | None):
        """
        Args:
            raw_path (str): Path to the raw dataset file.
            output_dir (str | None): Directory where processed outputs will be saved. 
        """
        self.raw_path = raw_path
        self.output_dir = output_dir

        # Will hold standardized dataframe after processing
        self.df_standard: pd.DataFrame | None = None

        # Standardized dataframe columns order
        self.order_df_standar = ["sequence_id", "mutation", "ddg"]

        # Will hold dictionary mapping sequence_id -> sequence string
        self.sequences: dict[str, str] | None = None

    @abstractmethod
    def load_raw(self) -> pd.DataFrame:
        """
        Load the raw dataset from file.
        Must be implemented by subclasses.

        Returns:
            pd.DataFrame: Raw dataset as a DataFrame.
        """
        pass

    @abstractmethod
    def process(self) -> None:
        """
        Process the raw dataset into standardized form.
        Must be implemented by subclasses.

        Returns:
            None
        """
        pass


    def write_fasta(self, sequences: dict[str, str], fasta_out: str) -> None:
        """
        Write sequences to a FASTA file.

        Args:
            sequences (dict[str, str]): Mapping from sequence ID to sequence string.
            fasta_out (str): Output FASTA file path.
        Returns:
            None
        """
        records = [
            SeqRecord(Seq(seq), id=seq_id, description="") 
            for seq_id, seq in sequences.items()
        ]
        SeqIO.write(records, fasta_out, "fasta")

    def save_outputs(self, df_filename: str = "mut_data.csv", fasta_filename: str = "wt_sequences.fasta") -> None:
        """
        Save processed dataset outputs to CSV and FASTA files.

        Args:
            df_filename (str, optional): Name of the output CSV file. Defaults to "mut_data.csv".
            fasta_filename (str, optional): Name of the output FASTA file. Defaults to "wt_sequences.fasta".
        Returns:
            None
        """
        os.makedirs(self.output_dir, exist_ok=True)

        # Save standardized dataframe
        df_out = os.path.join(self.output_dir, df_filename)
        self.df_standard.to_csv(df_out, index=False)

        # Save sequences in FASTA format
        fasta_out = os.path.join(self.output_dir, fasta_filename)
        self.write_fasta(self.sequences, fasta_out)



class Loader1(BaseLoader):
    """
    Loader for dataset type '1'.
    Reads from Excel, standardizes columns, and fetches sequences from UniProt.
    """

    def load_raw(self) -> pd.DataFrame:
        """
        Load raw dataset from an Excel file.

        Returns:
            pd.DataFrame: Raw dataset loaded from the Excel file.
        """
        return pd.read_excel(self.raw_path)

    def process(self) -> None:
        """
        Process the raw dataset into standardized format:
        - Renames columns to ["sequence_id", "mutation", "ddg"]
        - Fetches protein sequences for all sequence IDs
        Saves results into `self.df_standard` and `self.sequences`.

        Returns:
            None
        """
        df = self.load_raw()

        # Standardize column names
        df_standard = df.rename(columns={
            "uniprot": "sequence_id",
            "mut": "mutation",
            "ddg": "ddg"
        })
        df_standard = df_standard[self.order_df_standar]

        self.df_standard = df_standard
        self.sequences = self.fetch_sequences()

    def fetch_sequences(self) -> dict[str, str]:
        """
        Fetch protein sequences for all unique sequence IDs in the dataset.

        Returns:
            dict[str, str]: Mapping of sequence IDs to their sequences.
        """
        resolver = SequenceResolver(db_name='uniprot')
        sequence_ids = self.df_standard["sequence_id"].unique().tolist()

        print('Fetching sequences (might take a while)...')
        sequences = {seq_id: resolver.fetch_sequence(seq_id) for seq_id in sequence_ids}

        return sequences
