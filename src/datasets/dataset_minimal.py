"""
Module: MinimalDataset
Description: Minimal dataset handler for mutation data without full metadata.
"""

from .dataset_base import MutationDataset
from .types import MutationSample
import pandas as pd


class MinimalDataset(MutationDataset):

    def __init__(self, csv_file, mode="train"):
        """
        Initialize Minimal dataset from CSV file.
        
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
            f"minimal_{i:06d}" for i in range(len(self.data))
        ]

        if self.mode == "train" and "ddg" not in self.data.columns:
            raise ValueError(
                f"Training mode activated (mode='train'), "
                f"but couldn't find 'ddg' column in {csv_file}"
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
            if col == "uniprot"
        }

        return MutationSample(
            sample_id=row["sample_id"],
            wt_id=row["uniprot"],
            mutation=row["mutation"],
            sequence_wt=row["wt_sequence"],
            ddg=None if self.mode != "train" else float(row["ddg"]),
            metadata=metadata
        )
