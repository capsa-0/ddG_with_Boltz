# src/ddg_predictor/data_processing/loaders.py

import os
import logging
import pandas as pd
import requests
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from tqdm import tqdm

class SequenceResolver:
    """
    Utility class to resolve biological sequences by ID from UniProt.
    """
    def fetch_uniprot_sequence(self, uniprot_id: str) -> str | None:
        """
        Fetches a sequence directly from UniProt REST API.
        """
        url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
        try:
            r = requests.get(url, timeout=10)
            r.raise_for_status()
            fasta_lines = r.text.splitlines()
            return "".join(fasta_lines[1:])
        except requests.RequestException as e:
            # Use logging for warnings instead of printing
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

def prepare_dataset(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Processes the raw dataframe into a standardized format and fetches sequences.
    """
    df = raw_df.rename(columns={
        "uniprot": "sequence_id",
        "mut": "mutation",
        "ddg": "ddg"
    })
    
    required_cols = ["sequence_id", "mutation", "ddg"]
    if not all(col in df.columns for col in required_cols):
        raise KeyError(f"Dataframe must contain the columns: {required_cols}")
    
    df_standard = df[required_cols]

    resolver = SequenceResolver()
    sequence_ids = df_standard["sequence_id"].unique().tolist()
    
    sequences = {}
    # tqdm provides progress without cluttering the log
    for seq_id in tqdm(sequence_ids, desc="Fetching protein sequences"):
        seq = resolver.fetch_uniprot_sequence(seq_id)
        if seq:
            sequences[seq_id] = seq
    
    # Filter out rows where sequence fetching failed
    valid_ids = sequences.keys()
    original_rows = len(df_standard)
    df_standard = df_standard[df_standard['sequence_id'].isin(valid_ids)].reset_index(drop=True)
    if len(df_standard) < original_rows:
        logging.warning(f"{original_rows - len(df_standard)} rows removed due to sequence fetching failures.")

    return df_standard, sequences

def save_prepared_data(df: pd.DataFrame, sequences: dict[str, str], config: dict):
    """
    Saves the standardized dataframe and sequences using filenames from the config.
    """
    output_dir = config['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    df_out_path = os.path.join(output_dir, config['mutations_csv_filename'])
    df.to_csv(df_out_path, index=False)
    logging.info(f"Mutation data saved to: {df_out_path}")

    fasta_out_path = os.path.join(output_dir, config['wt_fasta_filename'])
    records = [
        SeqRecord(Seq(seq), id=seq_id, description="") 
        for seq_id, seq in sequences.items()
    ]
    SeqIO.write(records, fasta_out_path, "fasta")
    logging.info(f"WT sequences saved to: {fasta_out_path}")

def load_prepare_save(config: dict):
    """
    Main orchestration function for the data loading and preparation step.
    """

    raw_df = load_raw_dataset(config['raw_data_path'])
    logging.info(f"Loaded {len(raw_df)} records from raw dataset.")

    df_standard, sequences = prepare_dataset(raw_df)
    
    save_prepared_data(df_standard, sequences, config)