# src/ddg_predictor/data_processing/loaders.py

import os
import logging
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm
import yaml
from pathlib import Path
import os
import shutil

class SequenceResolver:
    """
    Utility class to resolve biological sequences by ID from UniProt.
    """
    def fetch_uniprot_sequence(self, uniprot_id: str) -> str | None:
        """
        Fetches a sequence directly from the UniProt REST API.
        """
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            fasta_lines = r.text.splitlines()
            return "".join(fasta_lines[1:])
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch sequence for {uniprot_id}: {e}")
            return None

def load_raw_dataset(raw_path: str) -> pd.DataFrame:
    """
    Loads a raw dataset from an Excel or CSV file.
    """
    file_extension = os.path.splitext(raw_path)[1].lower()
    if file_extension in ['.xlsx', '.xls']:
        return pd.read_excel(raw_path)
    elif file_extension == '.csv':
        return pd.read_csv(raw_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}. Must be .xlsx, .xls, or .csv")

def prepare_dataset(raw_df: pd.DataFrame, dataset_type: str) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Processes the raw dataframe into a standardized format and fetches sequences.
    Behavior depends on the `dataset_type` ('standard' or 'fireprot').
    """
    if dataset_type == 'fireprot':
        # Logic for the FireProt format
        df = raw_df.rename(columns={
            "uniprot_id": "sequence_id",
            "ddG": "ddg",
            "mutation": "mutant_aa" # Rename to avoid column name collision
        })
        
        # Validate that the necessary columns to build the mutation string exist
        required_fireprot_cols = ["wild_type", "position", "mutant_aa"]
        if not all(col in df.columns for col in required_fireprot_cols):
            raise KeyError(f"FireProt format requires the columns: {required_fireprot_cols}")
        
        # Build the standard mutation column (e.g., A123C)
        df["mutation"] = df["wild_type"] + df["position"].astype(str) + df["mutant_aa"]
    
    elif dataset_type == 'standard':
        # Original logic for the standard format
        df = raw_df.rename(columns={
            "uniprot": "sequence_id",
            "mut": "mutation",
            "ddg": "ddg"
        })
    else:
        raise ValueError(f"Unsupported dataset type: '{dataset_type}'. Use 'standard' or 'fireprot'.")

    # --- The rest of the flow is common to both formats ---
    required_cols = ["sequence_id", "mutation", "ddg"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"The processed dataframe must contain the columns: {required_cols}")
    
    df_standard = df[required_cols]

    resolver = SequenceResolver()
    sequence_ids = df_standard["sequence_id"].unique().tolist()
    
    sequences = {}
    for seq_id in tqdm(sequence_ids, desc="Fetching protein sequences"):
        seq = resolver.fetch_uniprot_sequence(seq_id)
        if seq:
            sequences[seq_id] = seq
    
    valid_ids = sequences.keys()
    original_rows = len(df_standard)
    df_standard = df_standard[df_standard['sequence_id'].isin(valid_ids)].reset_index(drop=True)
    if len(df_standard) < original_rows:
        logging.warning(f"{original_rows - len(df_standard)} rows removed due to sequence fetching failures.")

    return df_standard, sequences


def save_prepared_data(df: pd.DataFrame, sequences: dict[str, str], config: dict):
    """
    Saves the standardized dataframe, sequences, and a copy of the config YAML to the output directory.
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Save mutation data CSV
    df_out_path = os.path.join(output_dir, config['mutations_csv_filename'])
    df.to_csv(df_out_path, index=False)
    logging.info(f"Mutation data saved to: {df_out_path}")

    # Save WT FASTA sequences
    fasta_out_path = os.path.join(output_dir, config['wt_fasta_filename'])
    records = [
        SeqRecord(Seq(seq), id=seq_id, description="") 
        for seq_id, seq in sequences.items()
    ]
    SeqIO.write(records, fasta_out_path, "fasta")
    logging.info(f"WT sequences saved to: {fasta_out_path}")

    # Copy config/params.yaml to output_dir
    source_config_path = Path("config/params.yaml")
    if source_config_path.exists():
        dest_config_path = os.path.join(output_dir, "params.yaml")
        shutil.copy(source_config_path, dest_config_path)
        logging.info(f"Copied config to: {dest_config_path}")
    else:
        logging.warning("config/params.yaml not found; skipping config copy.")


def load_prepare_save(config: dict):
    """
    Main orchestration function for the data loading and preparation step.
    """
    # Get the dataset type from config. If it doesn't exist, default to 'standard'.
    dataset_type = config.get('dataset_type', 'standard')
    logging.info(f"Processing dataset of type: '{dataset_type}'")

    raw_df = load_raw_dataset(config['raw_data_path'])
    logging.info(f"Loaded {len(raw_df)} records from raw dataset.")

    df_standard, sequences = prepare_dataset(raw_df, dataset_type)
    
    save_prepared_data(df_standard, sequences, config)

def load_config(config_path: str) -> None:
    """
    Loads configuration parameters from a YAML file.
    """
    # Cargar archivo YAML
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extraer el nombre base del dataset desde raw_data_path
    raw_data_file = Path(config["data_processing"]["raw_data_path"])
    dataset_name = raw_data_file.stem  # e.g., S11304_tiny

    # Crear rutas dinámicas
    output_dir = f"data/processed/{dataset_name}"
    msa_output_dir = f"{output_dir}/msas"
    queries_dir = f"{output_dir}/boltz_queries"
    out_dir = f"{dataset_name}_results"

    # Agregar campos derivados al YAML
    config["data_processing"]["output_dir"] = output_dir
    config["data_processing"]["msa_output_dir"] = msa_output_dir
    config["data_processing"]["queries_dir"] = queries_dir
    config["data_processing"]["wt_fasta_filename"] = "wt_sequences.fasta"
    config["data_processing"]["mutations_csv_filename"] = "mutations.csv"

    config["feature_extraction"].setdefault("boltz_flags", {})
    config["feature_extraction"]["boltz_flags"]["out_dir"] = out_dir

    # Guardar nuevo YAML
    with open("config/params.yaml", "w") as f:
        yaml.dump(config, f, sort_keys=False)