# src/ddg_predictor/data/datasets.py

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import yaml

def load_npz(path):
    """Helper function to load a tensor from a .npz file."""
    if not os.path.exists(path):
        # Return None if a file is missing, the collate function will handle it.
        return None
    with np.load(path) as data:
        return torch.from_numpy(data[data.files[0]]).float()

def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors by padding.
    """
    # Separate the different tensor types and labels from the batch
    pdist_list, s_list, z_list, ddg_list = [], [], [], []
    valid_samples = []

    for sample in batch:
        # Check if all tensors for a sample were loaded correctly
        if sample['pdistogram'] is not None and sample['s'] is not None and sample['z'] is not None:
            pdist_list.append(sample['pdistogram'])
            s_list.append(sample['s'])
            z_list.append(sample['z'])
            ddg_list.append(sample['ddg'])
            valid_samples.append(True)
        else:
            valid_samples.append(False)
    
    # If a whole batch is invalid, return None
    if not ddg_list:
        return None

    # Use pad_sequence for tensors that vary in the first dimension (N)
    # batch_first=True makes the output shape (Batch, N, ...)
    s_padded = pad_sequence(s_list, batch_first=True, padding_value=0.0)
    
    # For NxN tensors, we need a custom padding logic
    def pad_2d_tensors(tensors):
        max_n = max(t.shape[0] for t in tensors)
        channels = tensors[0].shape[2]
        padded = torch.zeros(len(tensors), max_n, max_n, channels)
        for i, t in enumerate(tensors):
            n = t.shape[0]
            padded[i, :n, :n, :] = t
        return padded

    pdist_padded = pad_2d_tensors(pdist_list)
    z_padded = pad_2d_tensors(z_list)

    # Stack the labels (ddg) as they are all the same size
    ddg_batch = torch.stack(ddg_list)

    return {
        "pdistogram": pdist_padded,
        "s": s_padded,
        "z": z_padded,
        "ddg": ddg_batch
    }


class DDGDataset(Dataset):
    def __init__(self, results_dir: str):
        """
        Initializes the dataset from a self-contained results directory.

        Args:
            results_dir (str): Path to the directory containing embeddings, 
                               mutations.csv, and params.yaml.
        """
        self.results_dir = results_dir
        
        # Load the params.yaml specific to this results directory
        params_path = os.path.join(results_dir, "params.yaml")
        if not os.path.exists(params_path):
            raise FileNotFoundError(f"params.yaml not found in results directory: {params_path}")
        with open(params_path, "r") as f:
            config = yaml.safe_load(f)
            
        dp_config = config['data_processing']
        
        # Construct the path to the mutations CSV within the results directory
        mutations_csv_path = os.path.join(
            self.results_dir, 
            dp_config['mutations_csv_filename']
        )
        
        if not os.path.exists(mutations_csv_path):
            raise FileNotFoundError(f"Mutations CSV not found at {mutations_csv_path}")
        
        self.metadata = pd.read_csv(mutations_csv_path)

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        sample_info = self.metadata.iloc[idx]
        wt_id = sample_info["sequence_id"]
        mutation = sample_info["mutation"]
        mut_id = f"{wt_id}_{mutation}"
        ddg_label = torch.tensor([sample_info["ddg"]], dtype=torch.float32)
        
        # The mutation subdirectories are inside the main results directory
        mutation_dir = os.path.join(self.results_dir, mut_id)
        
        pdist_tensor = load_npz(os.path.join(mutation_dir, "diff_pdistogram.npz"))
        s_tensor = load_npz(os.path.join(mutation_dir, "diff_s.npz"))
        z_tensor = load_npz(os.path.join(mutation_dir, "diff_z.npz"))

        return {
            "pdistogram": pdist_tensor, "s": s_tensor, "z": z_tensor, "ddg": ddg_label
        }