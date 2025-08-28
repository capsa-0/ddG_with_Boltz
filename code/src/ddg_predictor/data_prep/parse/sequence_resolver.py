from Bio import SeqIO
import requests


class SequenceResolver:
    """
    Utility class to resolve biological sequences by ID.
    Supports fetching from:
    - Local FASTA files (if provided)
    - UniProt REST API (if db_name='uniprot')
    """

    def __init__(self, db_name: str = 'uniprot', fasta_db_path: str | None = None):
        """
        Args:
            db_name (str, optional): Name of the sequence database (default: 'uniprot').
            fasta_db_path (str | None, optional): Path to a FASTA file used as a local database.
        """
        self.db_name = db_name
        self.fasta_index: dict[str, SeqIO.SeqRecord] = {}

        # If a FASTA file is provided, build an index for fast lookups
        if fasta_db_path:
            self.fasta_index = SeqIO.to_dict(SeqIO.parse(fasta_db_path, "fasta"))

    def fetch_sequence(self, sequence_id: str) -> str | None:
        """
        Fetch a biological sequence by its identifier.

        Args:
            sequence_id (str): Identifier of the sequence (e.g., UniProt ID).

        Returns:
            str | None: The sequence string if found, otherwise None.
        """
        # Try to resolve from local FASTA index first
        if sequence_id in self.fasta_index:
            return str(self.fasta_index[sequence_id].seq)

        # If configured to use UniProt, fetch remotely
        if self.db_name == 'uniprot':
            return self.fetch_uniprot_sequence(sequence_id)

        return None

    def fetch_uniprot_sequence(self, uniprot_id: str) -> str:
        """
        Fetch a sequence directly from UniProt REST API in FASTA format.

        Args:
            uniprot_id (str): UniProt sequence identifier.

        Returns:
            str: Sequence string (without FASTA header).
        """
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        r = requests.get(url)
        r.raise_for_status()  # Raise error if request fails

        fasta = r.text.splitlines()
        return "".join(fasta[1:])  # Skip the header line
