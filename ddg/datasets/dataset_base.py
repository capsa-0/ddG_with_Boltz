"""
Module: MutationDataset
Description: Abstract base class for mutation datasets with conversion utilities.
"""

from dataclasses import asdict
from abc import ABC, abstractmethod
import pandas as pd


class MutationDataset(ABC):

    @abstractmethod
    def __len__(self):
        """Return number of samples in dataset."""
        pass

    @abstractmethod
    def __getitem__(self, idx):
        """Return sample at given index."""
        pass

    def to_mutations_dataframe(self) -> pd.DataFrame:
        """
        Convert dataset to DataFrame with mutation information.
        Returns all samples without metadata column.
        """
        rows = []
        for i in range(len(self)):
            sample = self[i]
            sample_dict = asdict(sample)
            sample_dict.pop("metadata", None)
            rows.append(sample_dict)

        return pd.DataFrame(rows)

    def to_metadata_dataframe(self) -> pd.DataFrame:
        """
        Convert dataset to DataFrame with sample metadata.
        Returns sample_id and all metadata columns.
        """
        rows = []
        for i in range(len(self)):
            sample = self[i]
            row = {
                "sample_id": sample.sample_id,
                **sample.metadata
            }
            rows.append(row)

        return pd.DataFrame(rows)
