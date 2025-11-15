# src/ddg_predictor/feature_processing/feature_analyzer.py

import os
import re
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm

# Importación segura de matplotlib para visualización
try:
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
except ImportError:
    plt = None

class FeatureAnalyzer:
    """
    Responsibility: Read the pre-calculated 'diff' tensors, apply statistical 
    aggregators (mean, max, entropy, etc.) globally and locally (at mutation site),
    and create a tabular dataset (CSV).
    """
    
    def __init__(self, config: dict):
        self.dp_config = config['data_processing']
        self.fe_config = config['feature_extraction']
        
        self.embeddings_dir = self.fe_config['boltz_flags']['out_dir']
        self.mutations_csv_path = os.path.join(
            self.embeddings_dir,
            self.dp_config['mutations_csv_filename']
        )
        self.summary_output_dir = self.embeddings_dir

        # Definición de agregadores disponibles
        self.AGGREGATORS = {
            "mean": lambda arr: np.mean(arr),
            "std": lambda arr: np.std(arr),
            "max": lambda arr: np.max(arr),
            "sum": lambda arr: np.sum(arr),
            "entropy": lambda arr: self._calculate_entropy(arr),
            "gini": lambda arr: self._calculate_gini(arr),
            "mean_abs": lambda arr: np.mean(np.abs(arr)),
        }

    @staticmethod
    def _load_embedding(file_path: str) -> np.ndarray | None:
        if not os.path.exists(file_path):
            return None
        # np.squeeze elimina dimensiones 1. 
        # Ej: (4, 1, 382, 384) -> (4, 382, 384)
        with np.load(file_path) as data:
            return np.squeeze(data[data.files[0]])

    @staticmethod
    def _parse_mutation_pos(mutation_str: str) -> int:
        """
        Parses strings like 'A58M' to extract the position index.
        Returns 0-based index (e.g., 58 -> 57).
        """
        match = re.search(r'(\d+)', mutation_str)
        if match:
            return int(match.group(1)) - 1
        raise ValueError(f"Could not parse position from mutation string: {mutation_str}")

    def _get_local_slice(self, tensor: np.ndarray, pos: int, emb_type: str) -> np.ndarray:
        """
        Extracts the features corresponding strictly to the mutation site.
        - For 's' (L, D): Returns vector [pos, :]
        - For 'z'/'pdistogram' (L, L, D): Returns diagonal vector [pos, pos, :]
        """
        L = tensor.shape[0]
        if pos < 0 or pos >= L:
            logging.warning(f"Position {pos} out of bounds for length {L}.")
            return np.array([])

        if emb_type == "s":
            # Single representation: Tomamos el vector del residuo
            return tensor[pos, ...]
        else:
            # Pair representation: Tomamos la diagonal (interacción consigo mismo / estado interno)
            return tensor[pos, pos, ...]

    @staticmethod
    def _calculate_entropy(arr: np.ndarray) -> float:
        """Shannon entropy calculation."""
        arr = np.abs(arr).flatten()
        arr_sum = np.sum(arr)
        if arr_sum == 0: return 0.0
        
        probs = arr / arr_sum
        probs = probs[probs > 0]
        return -np.sum(probs * np.log(probs))

    @staticmethod
    def _calculate_gini(arr: np.ndarray) -> float:
        """Gini coefficient calculation."""
        arr = np.abs(arr).flatten()
        if np.sum(arr) == 0: return 0.0
        
        arr = np.sort(arr)
        n = arr.size
        index = np.arange(1, n + 1)
        return ((np.sum((2 * index - n - 1) * arr)) / (n * np.sum(arr)))

    def summarize_features(self, 
                           embedding_types=("s", "z", "pdistogram"), 
                           aggregations=("mean", "max", "entropy")) -> pd.DataFrame:
        
        if not os.path.exists(self.mutations_csv_path):
            raise FileNotFoundError(f"Mutations CSV not found at {self.mutations_csv_path}")

        mutations_df = pd.read_csv(self.mutations_csv_path)
        logging.info(f"Summarizing features for {len(mutations_df)} mutations.")
        
        results = []
        
        for _, row in tqdm(mutations_df.iterrows(), total=len(mutations_df), desc="Aggregating features"):
            wt_id, mutation = row["sequence_id"], row["mutation"]
            mut_id = f"{wt_id}_{mutation}"
            
            # Intentamos obtener la posición
            try:
                mut_pos = self._parse_mutation_pos(mutation)
            except ValueError:
                logging.warning(f"Skipping {mut_id} due to parse error.")
                continue

            result_row = row.to_dict()

            for emb_type in embedding_types:
                diff_path = os.path.join(self.embeddings_dir, mut_id, f"diff_{emb_type}.npz")
                diff_emb = self._load_embedding(diff_path)
                
                # Si no existe el archivo, rellenamos con NaNs
                if diff_emb is None:
                    for region in ["global", "local"]:
                        for agg in aggregations:
                            result_row[f"{region}_{emb_type}_{agg}"] = np.nan
                    continue

                # --- Lógica de Reciclado (Recycling Steps) ---
                # s squeezed: (4, 382, 384) -> ndim=3. Queremos el último (382, 384).
                # z squeezed: (4, 382, 382, 128) -> ndim=4. Queremos el último (382, 382, 128).
                # pdistogram squeezed: (4, 382, 382, 64) -> ndim=4. Queremos el último.
                
                final_emb = diff_emb
                
                # Si 's' tiene 3 dimensiones, la primera es steps (si fuera solo L,D serían 2)
                if emb_type == "s" and diff_emb.ndim == 3:
                    final_emb = diff_emb[-1]
                # Si 'z' o 'pdist' tienen 4 dimensiones, la primera es steps (si fuera L,L,D serían 3)
                elif emb_type in ["z", "pdistogram"] and diff_emb.ndim == 4:
                    final_emb = diff_emb[-1]

                # --- Extracción y Agregación ---
                
                # 1. Global: Todo el tensor procesado
                global_data = final_emb
                # 2. Local: Solo la posición de interés
                local_data = self._get_local_slice(final_emb, mut_pos, emb_type)
                
                slices = {
                    "global": global_data,
                    "local": local_data
                }

                for prefix, data_slice in slices.items():
                    # Si el slice está vacío (error de rango), saltamos
                    if data_slice.size == 0:
                        continue

                    for agg_name in aggregations:
                        col_name = f"{prefix}_{emb_type}_{agg_name}"
                        
                        func = self.AGGREGATORS.get(agg_name)
                        if func:
                            result_row[col_name] = func(data_slice)
            
            results.append(result_row)
        
        df_out = pd.DataFrame(results)
        output_filename = self.dp_config.get("summary_csv_filename", "features_summary.csv")
        out_path = os.path.join(self.summary_output_dir, output_filename)
        df_out.to_csv(out_path, index=False)
        
        logging.info(f"Feature summary saved to: {out_path}")
        return df_out

    def plot_feature_correlations(self, df: pd.DataFrame = None):
        """
        Generates scatter plots for each calculated feature against ddG.
        Creates a 'correlation_plots' folder in the summary directory.
        """
        if plt is None:
            logging.error("Matplotlib is not installed. Cannot generate plots.")
            return

        # Si no se pasa un DF, intenta cargar el que acabamos de generar
        if df is None:
            filename = self.dp_config.get("summary_csv_filename", "features_summary.csv")
            csv_path = os.path.join(self.summary_output_dir, filename)
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
            else:
                logging.error("No DataFrame provided and summary CSV not found.")
                return

        if "ddg" not in df.columns:
            logging.error("DataFrame does not contain 'ddg' column. Cannot plot correlations.")
            return

        # Directorio de salida para los plots
        plots_dir = os.path.join(self.summary_output_dir, "correlation_plots")
        os.makedirs(plots_dir, exist_ok=True)

        # Identificar columnas de features (empiezan por 'global_' o 'local_')
        feature_cols = [c for c in df.columns if c.startswith(('global_', 'local_'))]
        
        # Preparar colores únicos por proteína
        if "sequence_id" in df.columns:
            unique_prots = df["sequence_id"].unique()
            colors = plt.cm.jet(np.linspace(0, 1, len(unique_prots)))
            color_map = dict(zip(unique_prots, colors))
        else:
            color_map = None

        logging.info(f"Generating correlation plots for {len(feature_cols)} features...")

        for feat_col in tqdm(feature_cols, desc="Plotting"):
            # Limpieza de NaNs para el cálculo de correlación
            clean_df = df.dropna(subset=["ddg", feat_col])
            
            if clean_df.empty:
                continue

            # Calcular correlación de Pearson
            corr = clean_df["ddg"].corr(clean_df[feat_col])

            plt.figure(figsize=(8, 6))
            
            # Plotting con colores por proteína
            if color_map:
                for prot_id, group in clean_df.groupby("sequence_id"):
                    plt.scatter(group["ddg"], group[feat_col], 
                                label=prot_id, 
                                color=color_map.get(prot_id),
                                alpha=0.7, edgecolors='w')
                # Leyenda (puede ser muy grande si hay muchas proteínas, ajustamos tamaño)
                if len(unique_prots) <= 15:
                    plt.legend(title="Protein", bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                plt.scatter(clean_df["ddg"], clean_df[feat_col], alpha=0.7, edgecolors='w')

            plt.title(f"Correlation: {corr:.3f} | {feat_col} vs ddG")
            plt.xlabel("ddG (Experimental)")
            plt.ylabel(feat_col)
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()

            # Guardar
            out_name = f"scatter_{feat_col}.png"
            plt.savefig(os.path.join(plots_dir, out_name), dpi=150)
            plt.close() # Cerrar figura para liberar memoria

        logging.info(f"Plots saved to: {plots_dir}")