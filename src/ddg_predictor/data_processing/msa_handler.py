# src/ddg_predictor/data_processing/msa_handler.py

import os
import re
import glob
import logging
import random
import shutil
import tarfile
import time
from pathlib import Path
from typing import Optional, Union, Dict

import pandas as pd
import requests
import yaml
from Bio import SeqIO
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
logger = logging.getLogger(__name__)
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
TQDM_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"

# --- MMseqs2 API Wrapper (Internal) ---

def _run_mmseqs2(
    x: Union[str, list[str]],
    prefix: str = "tmp",
    use_env: bool = True,
    use_filter: bool = True,
    use_pairing: bool = False,
    pairing_strategy: str = "greedy",
    host_url: str = "https://api.colabfold.com",
    msa_server_username: Optional[str] = None,
    msa_server_password: Optional[str] = None,
    auth_headers: Optional[Dict[str, str]] = None,
) -> tuple[list[str], list[str]]:
    submission_endpoint = "ticket/pair" if use_pairing else "ticket/msa"

    # Validate mutually exclusive authentication methods
    has_basic_auth = msa_server_username and msa_server_password
    has_header_auth = auth_headers is not None
    if has_basic_auth and (has_header_auth or auth_headers):
        raise ValueError(
            "Cannot use both basic authentication (username/password) and header/API key authentication. "
            "Please use only one authentication method."
        )

    # Set header agent as boltz
    headers = {}
    headers["User-Agent"] = "boltz"

    # Set up authentication
    auth = None
    if has_basic_auth:
        auth = HTTPBasicAuth(msa_server_username, msa_server_password)
        logger.debug(f"MMSeqs2 server authentication: using basic auth for user '{msa_server_username}'")
    elif has_header_auth:
        headers.update(auth_headers)
        logger.debug("MMSeqs2 server authentication: using header-based authentication")
    else:
        logger.debug("MMSeqs2 server authentication: no credentials provided")
    
    logger.debug(f"Connecting to MMSeqs2 server at: {host_url}")
    logger.debug(f"Using endpoint: {submission_endpoint}")
    logger.debug(f"Pairing strategy: {pairing_strategy}")
    logger.debug(f"Use environment databases: {use_env}")
    logger.debug(f"Use filtering: {use_filter}")

    def submit(seqs, mode, N=101):
        n, query = N, ""
        for seq in seqs:
            query += f">{n}\n{seq}\n"
            n += 1

        error_count = 0
        while True:
            try:
                # https://requests.readthedocs.io/en/latest/user/advanced/#advanced
                # "good practice to set connect timeouts to slightly larger than a multiple of 3"
                logger.debug(f"Submitting MSA request to {host_url}/{submission_endpoint}")
                res = requests.post(
                    f"{host_url}/{submission_endpoint}",
                    data={"q": query, "mode": mode},
                    timeout=6.02,
                    headers=headers,
                    auth=auth,
                )
                logger.debug(f"MSA submission response status: {res.status_code}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                if error_count > 5:
                    raise Exception(
                        "Too many failed attempts for the MSA generation request."
                    )
                time.sleep(5)
            else:
                break

        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def status(ID):
        error_count = 0
        while True:
            try:
                logger.debug(f"Checking MSA job status for ID: {ID}")
                res = requests.get(
                    f"{host_url}/ticket/{ID}", timeout=6.02, headers=headers, auth=auth
                )
                logger.debug(f"MSA status check response status: {res.status_code}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                if error_count > 5:
                    raise Exception(
                        "Too many failed attempts for the MSA generation request."
                    )
                time.sleep(5)
            else:
                break
        try:
            out = res.json()
        except ValueError:
            logger.error(f"Server didn't reply with json: {res.text}")
            out = {"status": "ERROR"}
        return out

    def download(ID, path):
        error_count = 0
        while True:
            try:
                logger.debug(f"Downloading MSA results for ID: {ID}")
                res = requests.get(
                    f"{host_url}/result/download/{ID}", timeout=6.02, headers=headers, auth=auth
                )
                logger.debug(f"MSA download response status: {res.status_code}")
            except Exception as e:
                error_count += 1
                logger.warning(
                    f"Error while fetching result from MSA server. Retrying... ({error_count}/5)"
                )
                logger.warning(f"Error: {e}")
                if error_count > 5:
                    raise Exception(
                        "Too many failed attempts for the MSA generation request."
                    )
                time.sleep(5)
            else:
                break
        with open(path, "wb") as out:
            out.write(res.content)

    # process input x
    seqs = [x] if isinstance(x, str) else x

    # setup mode
    if use_filter:
        mode = "env" if use_env else "all"
    else:
        mode = "env-nofilter" if use_env else "nofilter"

    if use_pairing:
        mode = ""
        # greedy is default, complete was the previous behavior
        if pairing_strategy == "greedy":
            mode = "pairgreedy"
        elif pairing_strategy == "complete":
            mode = "paircomplete"
        if use_env:
            mode = mode + "-env"

    # define path
    path = f"{prefix}_{mode}"
    if not os.path.isdir(path):
        os.mkdir(path)

    # call mmseqs2 api
    tar_gz_file = f"{path}/out.tar.gz"
    N, REDO = 101, True

    # deduplicate and keep track of order
    seqs_unique = []
    # TODO this might be slow for large sets
    [seqs_unique.append(x) for x in seqs if x not in seqs_unique]
    Ms = [N + seqs_unique.index(seq) for seq in seqs]
    # lets do it!
    if not os.path.isfile(tar_gz_file):
        TIME_ESTIMATE = 150 * len(seqs_unique)
        with tqdm(total=TIME_ESTIMATE, bar_format=TQDM_BAR_FORMAT) as pbar:
            while REDO:
                pbar.set_description("SUBMIT")

                # Resubmit job until it goes through
                out = submit(seqs_unique, mode, N)
                while out["status"] in ["UNKNOWN", "RATELIMIT"]:
                    sleep_time = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {sleep_time}s. Reason: {out['status']}")
                    # resubmit
                    time.sleep(sleep_time)
                    out = submit(seqs_unique, mode, N)

                if out["status"] == "ERROR":
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

                if out["status"] == "MAINTENANCE":
                    msg = (
                        "MMseqs2 API is undergoing maintenance. "
                        "Please try again in a few minutes."
                    )
                    raise Exception(msg)

                # wait for job to finish
                ID, TIME = out["id"], 0
                logger.debug(f"MSA job submitted successfully with ID: {ID}")
                pbar.set_description(out["status"])
                while out["status"] in ["UNKNOWN", "RUNNING", "PENDING"]:
                    t = 5 + random.randint(0, 5)
                    logger.error(f"Sleeping for {t}s. Reason: {out['status']}")
                    time.sleep(t)
                    out = status(ID)
                    pbar.set_description(out["status"])
                    if out["status"] == "RUNNING":
                        TIME += t
                        pbar.update(n=t)

                if out["status"] == "COMPLETE":
                    logger.debug(f"MSA job completed successfully for ID: {ID}")
                    if TIME < TIME_ESTIMATE:
                        pbar.update(n=(TIME_ESTIMATE - TIME))
                    REDO = False

                if out["status"] == "ERROR":
                    REDO = False
                    msg = (
                        "MMseqs2 API is giving errors. Please confirm your "
                        " input is a valid protein sequence. If error persists, "
                        "please try again an hour later."
                    )
                    raise Exception(msg)

            # Download results
            download(ID, tar_gz_file)

    # prep list of a3m files
    if use_pairing:
        a3m_files = [f"{path}/pair.a3m"]
    else:
        a3m_files = [f"{path}/uniref.a3m"]
        if use_env:
            a3m_files.append(f"{path}/bfd.mgnify30.metaeuk30.smag30.a3m")

    # extract a3m files
    if any(not os.path.isfile(a3m_file) for a3m_file in a3m_files):
        with tarfile.open(tar_gz_file) as tar_gz:
            tar_gz.extractall(path)

    # gather a3m lines
    a3m_lines = {}
    for a3m_file in a3m_files:
        update_M, M = True, None
        for line in open(a3m_file, "r"):
            if len(line) > 0:
                if "\x00" in line:
                    line = line.replace("\x00", "")
                    update_M = True
                if line.startswith(">") and update_M:
                    M = int(line[1:].rstrip())
                    update_M = False
                    if M not in a3m_lines:
                        a3m_lines[M] = []
                a3m_lines[M].append(line)

    a3m_lines = ["".join(a3m_lines[n]) for n in Ms]




    if os.path.exists(path):
        shutil.rmtree(path)  



    return a3m_lines


# --- MSA Generation ---

class MsaGenerator:
    """Generates MSAs for wild-type sequences using an MMseqs2 server."""

    def __init__(self, output_dir: str, **mmseqs2_kwargs):
        self.output_dir = output_dir
        self.mmseqs2_kwargs = mmseqs2_kwargs
        os.makedirs(self.output_dir, exist_ok=True)

    def _get_sequences_from_fasta(self, fasta_path: str) -> dict[str, str]:
        """Parses a FASTA file into a dictionary of id -> sequence."""
        return {record.id: str(record.seq) for record in SeqIO.parse(fasta_path, "fasta")}

    def generate_for_fasta(self, fasta_path: str):
        """Generates MSAs for all sequences in a given FASTA file."""
        sequences = self._get_sequences_from_fasta(fasta_path)
        logging.info(f"Starting MSA generation for {len(sequences)} sequences.")

        for seq_id, sequence in tqdm(sequences.items(), desc="Generating WT MSAs"):
            output_path = os.path.join(self.output_dir, f"{seq_id}.a3m")
            if os.path.exists(output_path):
                logging.info(f"MSA for {seq_id} already exists. Skipping.")
                continue
            
            try:
                a3m_lines = _run_mmseqs2(x=sequence, prefix=f"tmp_{seq_id}", **self.mmseqs2_kwargs)
                msa_content = a3m_lines[0]
                
                # Replace generic header with sequence ID
                lines = msa_content.split('\n')
                if lines and lines[0].startswith('>'):
                    lines[0] = f'>{seq_id}'
                
                with open(output_path, 'w') as f:
                    f.write('\n'.join(lines))
            except Exception as e:
                logging.error(f"Failed to generate MSA for {seq_id}: {e}")

# --- MSA Mutation ---

class MsaMutator:
    """Aplica mutaciones puntuales a archivos A3M existentes."""

    def __init__(self, msa_dir: str, mutations_df: pd.DataFrame):
        self.msa_dir = msa_dir
        self.mutations_by_id = mutations_df.groupby("sequence_id")["mutation"].apply(list).to_dict()

    @staticmethod
    def _parse_mutation(mutation: str) -> tuple[str, int, str]:
        """Analiza una cadena de mutación como 'A23T'."""
        match = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation)
        if not match:
            raise ValueError(f"Formato de mutación inválido: {mutation}")
        orig_res, pos, new_res = match.groups()
        return orig_res, int(pos), new_res

    @staticmethod
    def _ungapped_sequence(seq_with_gaps: str) -> str:
        """Devuelve la secuencia sin gaps, solo con letras mayúsculas."""
        return "".join([aa for aa in seq_with_gaps if aa.isupper()])

    @staticmethod
    def _map_query_position_to_alignment(seq_with_gaps: str, query_pos: int) -> int:
        """Mapea una posición de la secuencia sin gaps a la posición en el alineamiento (base 1)."""
        count = 0
        for i, aa in enumerate(seq_with_gaps, start=1):
            if aa.isupper():
                count += 1
                if count == query_pos:
                    return i
        raise ValueError(f"La posición {query_pos} excede la longitud de la secuencia.")

    def _read_a3m(self, path: str) -> list[tuple[str, str]]:
        """Lee un archivo A3M y devuelve una lista de tuplas (header, secuencia)."""
        records = []
        header = None
        seq_parts = []
        with open(path, "r") as f:
            for line in f:
                line = line.rstrip()
                if line.startswith(">"):
                    if header is not None:
                        records.append((header, "".join(seq_parts)))
                    header = line
                    seq_parts = []
                else:
                    seq_parts.append(line.strip())
            if header is not None:
                records.append((header, "".join(seq_parts)))
        return records

    def _save_a3m(self, records: list[tuple[str, str]], path: str):
        """Guarda una lista de registros en un archivo A3M."""
        with open(path, "w") as f:
            for header, seq in records:
                f.write(f"{header}\n{seq}\n")

    def _apply_single_mutation(self, records: list, mutation: str) -> list[tuple[str, str]]:
        """Aplica una única mutación a los registros del MSA."""
        orig_res, query_pos, new_res = self._parse_mutation(mutation)
        

        query_header, query_seq = records[0]
        ungapped_query = self._ungapped_sequence(query_seq)

        if ungapped_query[query_pos - 1].upper() != orig_res:
            raise ValueError(f"El residuo original en la posición {query_pos} ('{ungapped_query[query_pos - 1]}') no coincide con la mutación ('{orig_res}').")

        aligned_pos = self._map_query_position_to_alignment(query_seq, query_pos)
        
        mutated_records = []

        original_id = Path(query_header.lstrip('>')).stem.split()[0]
        new_header = f">{original_id}_{mutation}"

        for i, (header, seq) in enumerate(records):
            mutated_seq = list(seq)
            if aligned_pos - 1 < len(mutated_seq):
                aa_at_pos = mutated_seq[aligned_pos - 1]
                if aa_at_pos != "-": 
                    new_char = new_res.lower() if aa_at_pos.islower() else new_res.upper()
                    mutated_seq[aligned_pos - 1] = new_char
            
            final_seq = "".join(mutated_seq)

            final_header = new_header if i == 0 else header
            mutated_records.append((final_header, final_seq))
        
        return mutated_records

    def mutate_all(self):
        """Itera sobre todos los MSAs WT y genera un MSA mutado para cada mutación listada."""
        logging.info(f"Aplicando mutaciones a los MSAs en {self.msa_dir}...")
        

        for seq_id, mutations in tqdm(self.mutations_by_id.items(), desc="Mutando MSAs"):
            wt_msa_path = os.path.join(self.msa_dir, f"{seq_id}.a3m")
            if not os.path.exists(wt_msa_path):
                logging.warning(f"MSA WT para {seq_id} no encontrado. Omitiendo sus mutaciones.")
                continue
            
     
            wt_records = self._read_a3m(wt_msa_path)
            
            for mutation in mutations:
                mut_msa_path = os.path.join(self.msa_dir, f"{seq_id}_{mutation}.a3m")
                if os.path.exists(mut_msa_path):
                    logging.info(f"MSA mutado {mut_msa_path} ya existe. Omitiendo.")
                    continue
                
                try:

                    mutated_records = self._apply_single_mutation(list(wt_records), mutation)
                    self._save_a3m(mutated_records, mut_msa_path)
                except ValueError as e:
                    logging.error(f"No se pudo aplicar la mutación {mutation} a {seq_id}: {e}")


# --- MSA Processing & Conversion ---
class IndentDumper(yaml.SafeDumper):
    """Custom YAML dumper for consistent indentation."""
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)

class MsaProcessor:
    """Provides tools for processing MSA files (truncating, converting)."""

    # --- Truncation Methods ---
    
    def _truncate_single_msa(self, msa_file_path: str, max_sequences: int):
        """Trunca un único archivo MSA de forma correcta."""
        with open(msa_file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            return 

        truncated_lines = []
        sequence_count = 0
        i = 0
        while i < len(lines):

            if lines[i].startswith('>'):

                if sequence_count < max_sequences:
                    sequence_count += 1
                    

                    truncated_lines.append(lines[i])
                    i += 1
                    
                    while i < len(lines) and not lines[i].startswith('>'):
                        truncated_lines.append(lines[i])
                        i += 1
                else:

                    break
            else:

                i += 1
                
        with open(msa_file_path, 'w') as f:
            f.writelines(truncated_lines)

    def truncate_msas_in_dir(self, msa_dir: str, max_sequences: int):
        """Truncates all A3M files in a directory."""
        msa_files = glob.glob(os.path.join(msa_dir, "*.a3m"))
        if not msa_files:
            logging.warning(f"No .a3m files found in {msa_dir} to truncate.")
            return
            
        logging.info(f"Truncating {len(msa_files)} MSA files to max {max_sequences} sequences.")
        for msa_file in tqdm(msa_files, desc="Truncating MSAs"):
            self._truncate_single_msa(msa_file, max_sequences)

    # --- A3M to YAML Conversion Methods ---

    def _extract_msa_query_info(self, a3m_file: str) -> tuple[str, str]:
        """Extracts query sequence ID and ungapped sequence from an A3M file."""
        with open(a3m_file, 'r') as f:
            lines = [line.strip() for line in f]
        
        header = lines[0][1:]
        ungapped_seq = re.sub(r'[\.\-]', '', lines[1])
        return header, ungapped_seq

    def _build_query_doc(self, seq_id: str, sequence: str, msa_path: str) -> dict:
        """Builds the dictionary structure for the Boltz YAML query."""
        return {
            "sequences": [{
                "protein": {
                    "id": seq_id,
                    "sequence": sequence,
                    "msa": os.path.abspath(msa_path)
                }
            }]
        }

    def convert_a3m_to_yaml(self, msa_dir: str, yaml_output_dir: str):
        """Converts all A3M files in a directory to Boltz-compatible YAML files."""
        os.makedirs(yaml_output_dir, exist_ok=True)
        a3m_files = sorted(glob.glob(os.path.join(msa_dir, "*.a3m")))
        
        if not a3m_files:
            logging.warning(f"No .a3m files found in {msa_dir} to convert to YAML.")
            return

        logging.info(f"Converting {len(a3m_files)} MSA files to YAML format...")

        for idx, a3m_path in enumerate(tqdm(a3m_files, desc="Converting A3M to YAML"), start=1):
            try:
                original_id, sequence = self._extract_msa_query_info(a3m_path)
                doc = self._build_query_doc(str(idx), sequence, a3m_path)
                

                safe_name = re.sub(r'[^\w\-\.]', '_', original_id)
                out_path = os.path.join(yaml_output_dir, f"{safe_name}.yaml")

                with open(out_path, 'w') as out_f:
                    out_f.write(f"# Original ID: {original_id}\n")
                    yaml.dump(
                        doc,
                        out_f,
                        Dumper=IndentDumper,
                        sort_keys=False,
                        default_flow_style=False,
                        indent=2,
                    )
            except Exception as e:
                logging.error(f"Failed to convert {os.path.basename(a3m_path)}: {e}")

# --- High-Level Orchestration Functions ---

def generate_wt_msas(fasta_path: str, output_dir: str, **mmseqs2_kwargs):
    """Orchestrates the generation of all wild-type MSAs."""
    print("--- Starting Step 2: Wild-Type MSA Generation ---")
    generator = MsaGenerator(output_dir, **mmseqs2_kwargs)
    generator.generate_for_fasta(fasta_path)
    print("--- Step 2 Finished ---")

def apply_mutations_to_msas(msa_dir: str, mutations_csv_path: str):
    """Orchestrates the creation of mutated MSAs based on a CSV file."""
    print("--- Starting Step 3: MSA Mutation ---")
    mutations_df = pd.read_csv(mutations_csv_path)
    mutator = MsaMutator(msa_dir, mutations_df)
    mutator.mutate_all()
    print("--- Step 3 Finished ---")

# al final de src/ddg_predictor/data_processing/msa_handler.py

def process_msas(msa_dir: str, yaml_output_dir: str, max_sequences: int | None = None):
    """
    Orchestrates post-processing of MSAs: truncation and conversion.
    """
    print("--- Starting MSA Post-Processing ---")
    processor = MsaProcessor()
    
    # Truncate all MSAs (WT and mutated) if max_sequences is specified
    if max_sequences:
        processor.truncate_msas_in_dir(msa_dir, max_sequences)
        
    # Convert all truncated MSAs to YAML queries for Boltz
    processor.convert_a3m_to_yaml(msa_dir, yaml_output_dir)
    print("--- MSA Post-Processing Finished ---")