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
    seq = "".join(fasta[1:])  
    return seq


# === MAIN ===

parser = argparse.ArgumentParser(description="Extract sequences from UniProt")
parser.add_argument("input_file", help="Excel file")
parser.add_argument("output_file", help="Output FASTA file")
args = parser.parse_args()


df = pd.read_excel(args.input_file)

mutaciones_por_proteina = defaultdict(list)


for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing mutations"):
    uniprot_id = row["uniprot"]
    mutation = row["mut"]
    ddg = float(str(row["ddg"]).replace(",", ".")) 

    seq = fetch_uniprot_sequence(uniprot_id)

    mutaciones_por_proteina[uniprot_id].append({
        "mutation": mutation,
        "ddg": ddg,
        "seq": seq
    })


output_file = args.output_file

os.makedirs(os.path.dirname(output_file), exist_ok=True)

with open(output_file, "w") as f_out:
    for uniprot_id, mut_list in mutaciones_por_proteina.items():

        mutaciones_str = "_".join(m["mutation"] for m in mut_list)
        header = f">{uniprot_id}_{mutaciones_str}"


        seq = mut_list[0]["seq"]


        f_out.write(header + "\n")
        f_out.write(seq + "\n")

