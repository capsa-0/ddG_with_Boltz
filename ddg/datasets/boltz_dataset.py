"""
Module: BoltzDataset
Description: PyTorch dataset for loading Boltz prediction embeddings and tensors.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
import logging

from ddg.datasets.ids import wt_key, mutant_key

logger = logging.getLogger(__name__)


class BoltzNPZLoader:
    """Loads NPZ files from Boltz predictions and extracts tensors."""
    
    @staticmethod
    def load_tensors(file_path: Path):
        """
        Load and process embedding tensors from Boltz NPZ file.
        
        Args:
            file_path: Path to NPZ file
            
        Returns:
            Tuple of (s_tensor, z_tensor, pdistogram_tensor)
            Handles multi-step recycling by extracting the last step
        """
        with np.load(file_path) as data:
            # Extract and convert to PyTorch tensors, squeeze singleton dimensions
            s_tensor = torch.from_numpy(np.squeeze(data['s']))
            z_tensor = torch.from_numpy(np.squeeze(data['z']))
            pdistogram = torch.from_numpy(np.squeeze(data['pdistogram']))
            
            # ----- Handle Recycling Steps -----
            # If multiple recycling steps: 's' is 3D (steps, L, D), take only last step
            if s_tensor.ndim == 3: 
                s_tensor = s_tensor[-1]
                
            # 'z' normally 3D (L, L, D), becomes 4D with steps, take last
            if z_tensor.ndim == 4: 
                z_tensor = z_tensor[-1]
                
            # pdistogram normally 3D (L, L, D), becomes 4D with steps, take last
            if pdistogram.ndim == 4: 
                pdistogram = pdistogram[-1]
            
        return s_tensor, z_tensor, pdistogram


class BoltzDataset(Dataset):
    """Maps Boltz output directory structure and serves embedding data."""
    
    def __init__(self, config):
        """
        Initialize dataset with experiment configuration.
        Builds index of available embedding files during initialization.
        
        Args:
            config: ProjectConfig instance with paths
        """
        self.df = pd.read_csv(config.mutations_df_path)
        output_dir = Path(config.raw_features_dir)
        self.predictions_dir = Path(output_dir) / "predictions"
        
        self.path_index = self._build_index()
        logger.debug(f"Initialized BoltzDataset with {len(self.path_index)} sequences")

    def _build_index(self):
        """
        Build index mapping sequence IDs to embedding file paths.
        Searches directory structure: predictions/ -> seq_id/ -> embeddings_seq_id.npz
        
        Returns:
            Dictionary mapping sequence ID to full path of embeddings NPZ file
        """
        index = {}
        for seq_folder in self.predictions_dir.iterdir():
            if seq_folder.is_dir():
                seq_id = seq_folder.name
                npz_path = seq_folder / f"embeddings_{seq_id}.npz"
                if npz_path.exists():
                    index[seq_id] = npz_path
        logger.debug(f"Built index with {len(index)} embedding files")
        return index

    def __len__(self) -> int:
        """Return number of mutation samples."""
        return len(self.df)

    def __getitem__(self, idx) -> dict:
        """
        Get embedding tensors for wild-type and mutant at given index.
        
        Args:
            idx: Sample index
            
        Returns:
            Dictionary with embedding tensors and metadata:
                - wt_id: Wild-type protein ID
                - mut_id: Mutation-specific ID
                - mutation: Mutation string
                - wt_s, wt_z, wt_pdistogram: Wild-type embeddings
                - mut_s, mut_z, mut_pdistogram: Mutant embeddings
                - ddg: Experimental ddG value
                
        Raises:
            FileNotFoundError: If embedding files not found
        """
        row = self.df.iloc[idx]
        wt_id = row['wt_id']
        mutation = row['mutation']

        # ----- Construct display identifier -----
        mut_id = f"{wt_id}_{mutation}"

        # ----- Look up embedding file paths using canonical (sanitized) keys -----
        # Prediction folders are named after the sanitized query header, so we
        # must sanitize wt_id/mutation the same way before indexing (bug 1.1).
        wt_lookup = wt_key(wt_id)
        mut_lookup = mutant_key(wt_id, mutation)
        wt_path = self.path_index.get(wt_lookup)
        mut_path = self.path_index.get(mut_lookup)

        if not wt_path or not mut_path:
            raise FileNotFoundError(
                f"Missing embeddings for {wt_id} or {mut_id} "
                f"(looked up keys '{wt_lookup}' / '{mut_lookup}')"
            )
            
        # ----- Load tensors using BoltzNPZLoader -----
        s_wt, z_wt, pdistogram_wt = BoltzNPZLoader.load_tensors(wt_path)
        s_mut, z_mut, pdistogram_mut = BoltzNPZLoader.load_tensors(mut_path)
        
        # ----- Return complete sample package -----
        return {
            "wt_id": wt_id,                           
            "mut_id": mut_id,
            "mutation": mutation,              
            "wt_s": s_wt, 
            "wt_z": z_wt, 
            "wt_pdistogram": pdistogram_wt,
            "mut_s": s_mut, 
            "mut_z": z_mut, 
            "mut_pdistogram": pdistogram_mut,
            "ddg": torch.tensor(row['ddg'], dtype=torch.float32)
        }