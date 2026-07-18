"""
Module: FireProtDataset
Description: Dataset handler for FireProt mutation data.
"""

from .dataset_base import MutationDataset
from .types import MutationSample
import pandas as pd

CORE_COLUMNS = ["sequence", "position", "wild_type", "mutation"]


class FireProtDataset(MutationDataset):

    def __init__(self, csv_file, mode="train"):
        """
        Initialize FireProt dataset from CSV file.
        
        Args:
            csv_file: Path to CSV file with mutation data
            mode: "train" or "inference" - whether to expect ddG labels
            
        Raises:
            ValueError: If training mode but ddG column missing
        """
        self.data = pd.read_csv(csv_file)
        self.mode = mode

        self.data = self.data.reset_index(drop=True)
        self.data["sample_id"] = [
            f"fireprot_{i:06d}" for i in range(len(self.data))
        ]
        
        if mode == "train" and "ddG" not in self.data.columns:
            raise ValueError(
                f"Training mode activated (mode='train'), "
                f"but couldn't find 'ddG' column in {csv_file}"
            )

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.data)

    def __getitem__(self, idx) -> MutationSample:
        """
        Get sample at given index.
        
        Args:
            idx: Sample index
            
        Returns:
            MutationSample with mutation data and metadata
        """
        row = self.data.iloc[idx]

        metadata = {
            col: row[col]
            for col in self.data.columns
            if col not in CORE_COLUMNS and col != "ddG"
        }

        return MutationSample(
            sample_id=row["sample_id"],
            wt_id=self.get_wt_id(row),
            mutation=self.get_mutation(row),
            sequence_wt=row["sequence"],
            ddg=None if self.mode != "train" else float(row["ddG"]),
            metadata=metadata
        )

    def get_wt_id(self, row) -> str:
        """
        Protein identifier used to key MSAs/queries and to group by protein.

        FireProt's primary key is ``uniprot_id``, but some entries (e.g. ThreeFoil
        / 3PG0, 2IMM, 1YYX) have no UniProt mapping while still carrying a valid
        ``pdb_id`` + sequence. Falling back to ``pdb_id`` keeps those proteins in
        the corpus instead of silently dropping every mutation with a NaN wt_id.
        """
        for col in ("uniprot_id", "pdb_id"):
            val = row.get(col)
            if isinstance(val, str) and val.strip():
                return val
        return row["uniprot_id"]  # NaN -> handled/dropped downstream as before

    def get_mutation(self, row) -> str:
        """Parse mutation string from row."""
        return f"{row['wild_type']}{row['position']}{row['mutation']}"