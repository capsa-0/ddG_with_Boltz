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
        raise ValueError(f"Could not retrieve sequence for {uniprot_id}")
    fasta = r.text.splitlines()
    seq = "".join(fasta[1:])
    return seq

parser = argparse.ArgumentParser(description="Extract sequences from UniProt")
parser.add_argument("input_file", help="Excel file")
parser.add_argument("output_dir", help="Base output directory")
args = parser.parse_args()

df = pd.read_excel(args.input_file)

mutations_by_protein = defaultdict(list)


for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing mutations"):
    uniprot_id = row["uniprot"]
    mutation = row["mut"]
    ddg = float(str(row["ddg"]).replace(",", "."))

    seq = fetch_uniprot_sequence(uniprot_id)

    mutations_by_protein[uniprot_id].append({
        "mutation": mutation,
        "ddg": ddg,
        "seq": seq
    })


output_dir = os.path.join(args.output_dir, "fasta_files")
os.makedirs(output_dir, exist_ok=True)

for uniprot_id, mut_list in mutations_by_protein.items():
    filename = os.path.join(output_dir, f"{uniprot_id}.fasta")
    mutations_str = "_".join(m["mutation"] for m in mut_list)
    header = f">{uniprot_id}_{mutations_str}"
    seq = mut_list[0]["seq"]

    with open(filename, "w") as f:
        f.write(header + "\n")
        f.write(seq + "\n")
