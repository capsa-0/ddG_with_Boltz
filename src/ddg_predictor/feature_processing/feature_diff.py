# src/ddg_predictor/feature_processing/feature_diff.py

import os
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

# (Matplotlib es una dependencia opcional para la visualización)
try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# src/ddg_predictor/feature_processing/feature_diff.py


class FeatureDiffer:
    """
    Calculates and processes the difference ('delta') between WT and mutant embeddings.
    """
    # --- DICCIONARIO DE AGREGADORES ---
    # Aquí definimos todas las funciones que podemos usar para resumir un tensor.
    # Es muy fácil añadir nuevas funciones aquí.
    AGGREGATORS = {
        "entropy": lambda arr: FeatureDiffer._calculate_entropy(arr),
        "gini": lambda arr: FeatureDiffer._calculate_gini(arr),
        "sum": lambda arr: np.sum(np.abs(arr)),
        "mean": lambda arr: np.mean(np.abs(arr)),
        "max": lambda arr: np.max(np.abs(arr)),
        "std": lambda arr: np.std(np.abs(arr)),
    }

    def __init__(self, config: dict):
        self.dp_config = config['data_processing']
        self.fe_config = config['feature_extraction']
        self.embeddings_dir = self.fe_config['boltz_flags']['out_dir']
        self.mutations_csv_path = os.path.join(
            self.embeddings_dir,
            self.dp_config['mutations_csv_filename']
        )
        self.summary_output_dir = self.embeddings_dir

    @staticmethod
    def _load_embedding(file_path: str) -> np.ndarray | None:
        if not os.path.exists(file_path):
            logging.warning(f"Embedding file not found: {file_path}")
            return None
        with np.load(file_path) as data:
            return np.squeeze(data[data.files[0]])

    def generate_diffs(self, embedding_types=("z", "s", "pdistogram")):
        """
        Calculates and saves the absolute difference between WT and mutant embeddings
        for all mutations defined in the mutations CSV.
        """
        if not os.path.exists(self.mutations_csv_path):
            raise FileNotFoundError(f"Mutations CSV not found at {self.mutations_csv_path}")
            
        mutations_df = pd.read_csv(self.mutations_csv_path)
        logging.info(f"Generating difference embeddings for {len(mutations_df)} mutations.")

        for _, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Calculating embedding diffs"):
            wt_id = row["sequence_id"]
            mutation = row["mutation"]
            mut_id = f"{wt_id}_{mutation}"

            for emb_type in embedding_types:
                # Construye la ruta al embedding de la proteína original (ej: .../P01308/s.npz)
                wt_path = os.path.join(self.embeddings_dir, wt_id, f"{emb_type}.npz")
                # Construye la ruta al embedding de la proteína mutada (ej: .../P01308_A2L/s.npz)
                mut_path = os.path.join(self.embeddings_dir, mut_id, f"{emb_type}.npz")

                wt_emb = self._load_embedding(wt_path)
                mut_emb = self._load_embedding(mut_path)

                # Si alguno de los dos no existe, salta al siguiente tipo de embedding
                if wt_emb is None or mut_emb is None:
                    continue
                
                # Comprobación de seguridad por si las dimensiones no coinciden
                if wt_emb.shape != mut_emb.shape:
                    logging.warning(f"Shape mismatch for {emb_type} in {mut_id}. WT: {wt_emb.shape}, MUT: {mut_emb.shape}. Skipping.")
                    continue

                # Calcula la diferencia absoluta
                diff = np.abs(mut_emb - wt_emb)
                
                # Guarda el archivo de diferencia en la carpeta de la mutación
                out_dir = os.path.join(self.embeddings_dir, mut_id)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"diff_{emb_type}.npz")
                np.savez(out_path, diff)

    @staticmethod
    def _calculate_entropy(arr: np.ndarray) -> float:
        arr = np.abs(arr).flatten()
        arr_sum = np.sum(arr)
        if arr_sum == 0: return 0.0
        arr = arr / arr_sum
        arr = arr[arr > 0]
        return -np.sum(arr * np.log(arr))

    @staticmethod
    def _calculate_gini(arr: np.ndarray) -> float:
        """Calculates the Gini coefficient of a numpy array."""
        arr = np.abs(arr).flatten()
        if np.sum(arr) == 0: return 0.0
        arr = np.sort(arr)
        n = arr.size
        coef = 2. * np.arange(1, n + 1).sum() - (n + 1)
        return np.sum(coef * arr) / (n * np.sum(arr))

    def calculate_deltas_summary(self, 
                                 embedding_types=("z", "s", "pdistogram"), 
                                 aggregations=("entropy", "sum", "max")) -> pd.DataFrame:
        """
        Calculates multiple summary statistics from the diff embeddings.
        """
        if not os.path.exists(self.mutations_csv_path):
            raise FileNotFoundError(f"Mutations CSV not found at {self.mutations_csv_path}")

        mutations_df = pd.read_csv(self.mutations_csv_path)
        logging.info(f"Calculating summary statistics for {len(mutations_df)} mutations using aggregators: {aggregations}")
        
        results = []
        for _, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Summarizing diffs"):
            wt_id, mutation = row["sequence_id"], row["mutation"]
            mut_id = f"{wt_id}_{mutation}"
            
            result_row = {"sequence_id": wt_id, "mutation": mutation, "ddg": row["ddg"]}

            for emb_type in embedding_types:
                diff_path = os.path.join(self.embeddings_dir, mut_id, f"diff_{emb_type}.npz")
                diff_emb = self._load_embedding(diff_path)
                
                # Para cada tipo de embedding, calcula todas las agregaciones solicitadas
                for agg_name in aggregations:
                    col_name = f"delta_{emb_type}_{agg_name}"
                    value = np.nan
                    if diff_emb is not None:
                        # Toma el último recycling step si aplica
                        final_step_emb = diff_emb[-1] if diff_emb.ndim > 2 else diff_emb
                        # Busca la función en el diccionario y la aplica
                        agg_func = self.AGGREGATORS.get(agg_name)
                        if agg_func:
                            value = agg_func(final_step_emb)
                        else:
                            logging.warning(f"Aggregator '{agg_name}' not found.")
                    
                    result_row[col_name] = value
            
            results.append(result_row)
        
        df_out = pd.DataFrame(results)
        filename = self.dp_config.get("summary_csv_filename", "features_summary.csv")
        out_path = os.path.join(self.summary_output_dir, filename)
        df_out.to_csv(out_path, index=False)
        logging.info(f"Feature summary saved to: {out_path}")
        return df_out

# --- Standalone Visualization Utility ---

def plot_delta_embeddings_for_mutation(mutation_dir: str, embedding_types=("pdistogram", "s", "z")):
    """
    Generates a figure with the delta embeddings for a single mutation.
    """
    if plt is None:
        logging.error("Matplotlib is not installed. Cannot generate plots.")
        return

    # (Aquí iría la lógica de tu función 'plot_delta_embeddings', refactorizada
    # para ser más limpia y tomar 'mutation_dir' como entrada.
    # Es principalmente una función de visualización, separada de la lógica de cálculo).
    logging.info(f"Generating plot for mutation at: {mutation_dir}")
    # ... Lógica de ploteo ...
    pass