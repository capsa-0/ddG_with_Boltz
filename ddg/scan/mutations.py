"""
Module: mutations
Description: Enumerate every possible single point mutation of a protein sequence.

A "full scan" of a length-L protein is L x 19 mutations: at each position, each of
the 19 standard amino acids other than the wild-type one. The output frame uses the
column names the ``minimal`` dataset adapter expects (``uniprot``, ``mutation``,
``wt_sequence``), so a scan flows through the existing pipeline unchanged.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

STANDARD_AA = "ACDEFGHIKLMNPQRSTVWY"

# Display order for the scan matrix / heatmap rows: grouped by side-chain chemistry
# (aliphatic-aromatic, polar, special, positive, negative) rather than alphabetically,
# so chemically similar substitutions sit next to each other and read as bands.
AA_ORDER = "AVLIMFWY" "STNQ" "CGP" "KRH" "DE"
assert sorted(AA_ORDER) == sorted(STANDARD_AA), "AA_ORDER must cover the 20 standard AAs"


def clean_sequence(sequence: str) -> str:
    """Upper-case and validate a protein sequence; raise on anything non-standard."""
    seq = "".join(str(sequence).split()).upper()
    if not seq:
        raise ValueError("empty sequence")
    bad = sorted({c for c in seq if c not in STANDARD_AA})
    if bad:
        raise ValueError(
            f"sequence contains non-standard residue(s) {bad}; the pipeline only "
            f"handles the 20 standard amino acids (see ddg.datasets.prepare)"
        )
    return seq


def all_point_mutations(sequence: str, wt_id: str) -> pd.DataFrame:
    """
    Every single point mutation of ``sequence``, as a ``minimal``-adapter frame.

    Mutation strings are 1-based over the given sequence (``<WT><pos><MUT>``), which
    is the format ddg.datasets.prepare parses and validates. Rows are ordered by
    position, then by target residue alphabetically, so the table is stable across
    runs.

    Returns a frame with columns: uniprot, mutation, wt_sequence.
    """
    seq = clean_sequence(sequence)
    rows = [
        {"uniprot": wt_id, "mutation": f"{wt_aa}{pos}{mut_aa}", "wt_sequence": seq}
        for pos, wt_aa in enumerate(seq, start=1)
        for mut_aa in STANDARD_AA
        if mut_aa != wt_aa
    ]
    df = pd.DataFrame(rows)
    logger.info("scan: %d residues -> %d point mutations (+1 wild-type structure)",
                len(seq), len(df))
    return df


def read_fasta_sequence(path) -> tuple[str, str]:
    """Read the first record of a FASTA file, returning (header_id, sequence)."""
    header, chunks = None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    break
                header = line[1:].split()[0]
            elif line:
                chunks.append(line)
    if header is None or not chunks:
        raise ValueError(f"no FASTA record found in {path}")
    return header, clean_sequence("".join(chunks))
