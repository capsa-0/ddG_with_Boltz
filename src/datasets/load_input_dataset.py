"""
Module: load_input_dataset
Description: Dataset loader factory for different dataset types.
"""

from .dataset_fireprot import FireProtDataset
from .dataset_minimal import MinimalDataset


def load_dataset(dataset_type, csv_file, mode="train"):
    """
    Load dataset of specified type.
    
    Args:
        dataset_type: Type of dataset ("fireprot", "minimal")
        csv_file: Path to dataset CSV file
        mode: "train" or "inference" mode
        
    Returns:
        Dataset instance of specified type
        
    Raises:
        ValueError: If dataset_type is not supported
    """
    dataset_classes = {
        'fireprot': FireProtDataset,
        'minimal': MinimalDataset,
    }
    if dataset_type not in dataset_classes:
        raise ValueError(
            f"Unsupported dataset type: '{dataset_type}'. "
            f"Available types: {list(dataset_classes.keys())}"
        )
    return dataset_classes[dataset_type](csv_file, mode=mode)