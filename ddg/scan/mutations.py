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


def parse_positions(text: str, first_residue: int = 1, length: int | None = None):
    """
    Parse a position selection like ``"80,137,169-175"`` into 1-based sequence indices.

    Numbers are given in the *reported* numbering (``first_residue``), which is how a
    user refers to residues, and converted to the 1-based sequence index the pipeline
    works in. Raises if anything falls outside the sequence, so a typo or a
    wrong-numbering mistake fails loudly instead of silently scanning nothing.
    """
    offset = int(first_residue) - 1
    wanted: set[int] = set()
    for chunk in str(text).replace(" ", "").split(","):
        if not chunk:
            continue
        if "-" in chunk.lstrip("-"):
            lo, hi = chunk.split("-", 1)
            span = range(int(lo), int(hi) + 1)
        else:
            span = [int(chunk)]
        wanted.update(p - offset for p in span)
    if length is not None:
        bad = sorted(p + offset for p in wanted if not 1 <= p <= length)
        if bad:
            raise ValueError(
                f"position(s) {bad} are outside the sequence "
                f"({first_residue}-{first_residue + length - 1} in this numbering)")
    return sorted(wanted)


def positions_with_residue(sequence: str, residues: str) -> list[int]:
    """1-based indices whose wild-type residue is one of ``residues`` (e.g. "G")."""
    wanted = set(str(residues).upper())
    return [i for i, aa in enumerate(clean_sequence(sequence), start=1) if aa in wanted]


def all_point_mutations(sequence: str, wt_id: str,
                        positions: list[int] | None = None) -> pd.DataFrame:
    """
    Every single point mutation of ``sequence``, as a ``minimal``-adapter frame.

    Mutation strings are 1-based over the given sequence (``<WT><pos><MUT>``), which
    is the format ddg.datasets.prepare parses and validates. Rows are ordered by
    position, then by target residue alphabetically, so the table is stable across
    runs.

    ``positions`` (1-based sequence indices) restricts the scan to those sites, all
    19 substitutions each — for when the full L x 19 scan does not fit the compute
    budget. Default None scans every position.

    Returns a frame with columns: uniprot, mutation, wt_sequence.
    """
    seq = clean_sequence(sequence)
    sites = sorted(set(positions)) if positions is not None else range(1, len(seq) + 1)
    bad = [p for p in sites if not 1 <= p <= len(seq)]
    if bad:
        raise ValueError(f"position(s) {bad} outside the sequence (1-{len(seq)})")
    rows = [
        {"uniprot": wt_id, "mutation": f"{seq[pos-1]}{pos}{mut_aa}", "wt_sequence": seq}
        for pos in sites
        for mut_aa in STANDARD_AA
        if mut_aa != seq[pos - 1]
    ]
    df = pd.DataFrame(rows)
    logger.info("scan: %d of %d residues -> %d point mutations "
                "(+1 wild-type structure)", len(list(sites)), len(seq), len(df))
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
