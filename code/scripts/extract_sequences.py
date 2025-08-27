import pandas as pd
import requests
from tqdm import tqdm
import argparse
import os
from collections import defaultdict

def fetch_uniprot_sequence(uniprot_id):
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url)
    if r.status_code != 200:
        raise ValueError(f"Couldn't get sequence for {uniprot_id}")
    fasta = r.text.splitlines()
    return "".join(fasta[1:])

parser = argparse.ArgumentParser(description="Extract sequences from UniProt")
parser.add_argument("input_file", help="Excel file")
parser.add_argument("output_file", help="Output FASTA file")
args = parser.parse_args()

df = pd.read_excel(args.input_file)

mutations_per_protein = defaultdict(list)

unique_uniprot_ids = df["uniprot"].unique()
sequences = {}

for uniprot_id in tqdm(unique_uniprot_ids, desc="Fetching sequences"):
    try:
        sequences[uniprot_id] = fetch_uniprot_sequence(uniprot_id)
    except Exception as e:
        print(f"Warning: {e}")
        sequences[uniprot_id] = None

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing mutations"):
    uniprot_id = row["uniprot"]
    mutation = row["mut"]
    ddg = float(str(row["ddg"]).replace(",", ".")) 

    seq = sequences.get(uniprot_id)
    if seq is None:
        continue

    mutations_per_protein[uniprot_id].append({
        "mutation": mutation,
        "ddg": ddg,
        "seq": seq
    })

output_file = args.output_file
os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f_out:
    for uniprot_id, mut_list in mutations_per_protein.items():
        mutaciones_str = "_".join(m["mutation"] for m in mut_list)
        header = f">{uniprot_id}_{mutaciones_str}"
        seq = mut_list[0]["seq"]
        f_out.write(header + "\n")
        f_out.write(seq + "\n")
