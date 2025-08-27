#!/usr/bin/env python3
import sys
import re
from pathlib import Path

def parse_mutation(mutation):

    m = re.match(r'^([A-Z])(\d+)([A-Z])$', mutation)

    a0, pos, a1 = m.groups()
    return a0, int(pos), a1

def read_a3m(path):
    recs = []
    head = None
    seq_parts = []
    with open(path) as f:
        for line in f:
            if line.startswith(">"):
                if head is not None:
                    recs.append((head, "".join(seq_parts)))
                head = line.rstrip()
                seq_parts = []
            else:
                seq_parts.append(line.strip())
        if head is not None:
            recs.append((head, "".join(seq_parts)))

    return recs

def write_a3m(path, records):
    with open(path, "w") as f:
        for h, s in records:
            f.write(f"{h}\n{s}\n")

def ungapped_query_aligned(seq_with_gaps):

    return "".join([aa for aa in seq_with_gaps if aa.isupper()])

def map_query_to_alignment(seq_with_gaps, pos_query):

    count = 0
    for i, aa in enumerate(seq_with_gaps, start=1):
        if aa.isupper():  
            count += 1
            if count == pos_query:
                return i


def mutar_a3m(input_file, output_file, mutation):
    res_orig, pos_query, res_new = parse_mutation(mutation)

    records = read_a3m(input_file)
    query_seq = records[0][1]

    ungapped = ungapped_query_aligned(query_seq)

    aa_at_query = ungapped[pos_query - 1].upper()


    pos_aln = map_query_to_alignment(query_seq, pos_query)

    mutated = []
    uniprot_id = Path(input_file).stem.split("_")[0]

    new_query_header = f">{uniprot_id}_{mutation}"

    for i, (header, seq) in enumerate(records):
        if pos_aln - 1 < len(seq):
            c = seq[pos_aln - 1]
            if c != "-":
                new_c = res_new.lower() if c.islower() else res_new.upper()
                seq = seq[:pos_aln - 1] + new_c + seq[pos_aln:]

        if i == 0:
            mutated.append((new_query_header, seq))
        else:
            mutated.append((header, seq))

    write_a3m(output_file, mutated)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(1)
    mutar_a3m(sys.argv[1], sys.argv[2], sys.argv[3])
