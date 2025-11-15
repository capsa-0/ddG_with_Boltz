# src/ddg_predictor/feature_processing/embedding_differ.py

import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

class EmbeddingDiffer:
    """
    Responsibility: Load raw Boltz embeddings (WT and Mutant), compute the 
    tensor difference (signed, absolute, etc.), and save the 'diff' tensors to disk.
    """
    def __init__(self, config: dict):
        self.dp_config = config['data_processing']
        self.fe_config = config['feature_extraction']
        
        # Rutas
        self.embeddings_dir = self.fe_config['boltz_flags']['out_dir']
        self.mutations_csv_path = os.path.join(
            self.embeddings_dir,
            self.dp_config['mutations_csv_filename']
        )

    @staticmethod
    def _load_embedding(file_path: str) -> np.ndarray | None:
        if not os.path.exists(file_path):
            logging.warning(f"Embedding file not found: {file_path}")
            return None
        with np.load(file_path) as data:
            # np.squeeze elimina dimensiones de tamaño 1 innecesarias
            return np.squeeze(data[data.files[0]])

    def _compute_diff(self, wt_emb: np.ndarray, mut_emb: np.ndarray, mode: str = "abs") -> np.ndarray:
        """Calcula la diferencia según el modo especificado."""
        if mode == "abs":
            return np.abs(mut_emb - wt_emb)
        elif mode == "signed":
            return mut_emb - wt_emb
        elif mode == "l2":
            # Norma L2 a lo largo de la última dimensión (útil para embeddings vectoriales como 's' o 'z')
            return np.linalg.norm(mut_emb - wt_emb, axis=-1)
        else:
            raise ValueError(f"Unknown diff mode: {mode}")

    def generate_diffs(self, embedding_types=("z", "s", "pdistogram"), mode="abs"):
        """
        Iterates over the mutations CSV, loads corresponding embeddings, 
        calculates differences, and saves diff_*.npz files.
        """
        if not os.path.exists(self.mutations_csv_path):
            raise FileNotFoundError(f"Mutations CSV not found at {self.mutations_csv_path}")
            
        mutations_df = pd.read_csv(self.mutations_csv_path)
        logging.info(f"Generating '{mode}' difference embeddings for {len(mutations_df)} mutations.")

        for _, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Computing diffs"):
            wt_id = row["sequence_id"]
            mutation = row["mutation"]
            mut_id = f"{wt_id}_{mutation}"

            for emb_type in embedding_types:
                # Construcción de rutas
                wt_path = os.path.join(self.embeddings_dir, wt_id, f"{emb_type}.npz")
                mut_path = os.path.join(self.embeddings_dir, mut_id, f"{emb_type}.npz")

                wt_emb = self._load_embedding(wt_path)
                mut_emb = self._load_embedding(mut_path)

                if wt_emb is None or mut_emb is None:
                    continue
                
                # Validación de dimensiones
                if wt_emb.shape != mut_emb.shape:
                    logging.warning(f"Shape mismatch for {emb_type} in {mut_id}. WT: {wt_emb.shape}, MUT: {mut_emb.shape}. Skipping.")
                    continue

                # Cálculo y guardado
                try:
                    diff = self._compute_diff(wt_emb, mut_emb, mode=mode)
                    
                    out_dir = os.path.join(self.embeddings_dir, mut_id)
                    os.makedirs(out_dir, exist_ok=True)
                    
                    out_path = os.path.join(out_dir, f"diff_{emb_type}.npz")
                    np.savez(out_path, diff)
                except Exception as e:
                    logging.error(f"Error processing {mut_id} ({emb_type}): {str(e)}")