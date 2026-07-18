"""
Module: cluster
Description: Build a {protein_id -> cluster_id} map for the homology holdout.

Clusters the unique WT sequences at a sequence-identity threshold with MMseqs2's
``easy-cluster`` (needs the ``mmseqs`` binary on PATH). If you already have a
cluster assignment, load it from CSV with ``load_cluster_map`` instead — the
benchmark only needs the dict.

Produce the map once and pass it to run_benchmark(..., cluster_map=...); it does
not touch the Boltz run.
"""

import csv
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def load_cluster_map(csv_path) -> dict:
    """Load a {protein_id: cluster_id} map from a 2-column CSV (id,cluster)."""
    out = {}
    with open(csv_path) as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        # tolerate presence/absence of a header row
        if header and not header[0].lower().startswith(("protein", "id", "wt")):
            out[header[0]] = header[1]
        for row in reader:
            if len(row) >= 2:
                out[row[0]] = row[1]
    logger.info("loaded %d protein->cluster assignments from %s", len(out), csv_path)
    return out


def cluster_by_identity(seqs: dict, threshold: float = 0.3, out_csv=None) -> dict:
    """
    Cluster sequences by pairwise identity with single linkage — no external
    binary, uses Biopython. Two sequences join the same cluster when their global
    (BLOSUM62) alignment identity, counted as identical residues / min(len), is
    >= ``threshold``; linkage is transitive (A~B, B~C => one cluster).

    For the ~400 short Tsuboyama domains the full O(N^2) sweep is a few seconds.
    Returns {id -> cluster_representative}. Prefer MMseqs2 (cluster_wt_sequences)
    when the binary is available and you want the canonical --min-seq-id semantics.
    """
    from Bio.Align import PairwiseAligner, substitution_matrices

    ids = list(seqs)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    aligner = PairwiseAligner(
        mode="global", substitution_matrix=substitution_matrices.load("BLOSUM62"),
        open_gap_score=-11, extend_gap_score=-1,
    )

    def identity(a, b):
        aln = aligner.align(a, b)[0]
        idn = 0
        for (a0, a1), (b0, b1) in zip(aln.aligned[0], aln.aligned[1]):
            idn += sum(1 for x, y in zip(a[a0:a1], b[b0:b1]) if x == y)
        return idn / min(len(a), len(b))

    for i in range(len(ids)):
        si = seqs[ids[i]]
        for j in range(i + 1, len(ids)):
            if find(ids[i]) == find(ids[j]):
                continue
            if identity(si, seqs[ids[j]]) >= threshold:
                union(ids[i], ids[j])

    mapping = {i: find(i) for i in ids}
    logger.info("identity clustering: %d seqs -> %d clusters at id>=%.2f",
                len(ids), len(set(mapping.values())), threshold)
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["protein_id", "cluster"])
            for k, v in mapping.items():
                w.writerow([k, v])
        logger.info("wrote cluster map to %s", out_csv)
    return mapping


def _read_fasta(fasta_path):
    seqs, name, buf = {}, None, []
    with open(fasta_path) as fh:
        for line in fh:
            line = line.rstrip()
            if line.startswith(">"):
                if name is not None:
                    seqs[name] = "".join(buf)
                name, buf = line[1:].strip(), []
            else:
                buf.append(line)
        if name is not None:
            seqs[name] = "".join(buf)
    return seqs


def cluster_wt_sequences(fasta_path, min_seq_id: float = 0.3,
                         coverage: float = 0.8, out_csv=None) -> dict:
    """
    Cluster the WT sequences in ``fasta_path`` at ``min_seq_id`` identity.

    Returns {header -> cluster_representative}. Writes a 2-column CSV if out_csv
    is given. Raises RuntimeError with instructions if mmseqs is not installed.
    """
    if shutil.which("mmseqs") is None:
        raise RuntimeError(
            "mmseqs not on PATH. Install it (conda install -c bioconda mmseqs2) "
            "or supply a precomputed map via load_cluster_map(). To build one "
            "manually:\n"
            f"  mmseqs easy-cluster {fasta_path} clust tmp "
            f"--min-seq-id {min_seq_id} -c {coverage}\n"
            "then use clust_cluster.tsv (cols: representative, member)."
        )
    fasta_path = Path(fasta_path)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        prefix = td / "clust"
        cmd = ["mmseqs", "easy-cluster", str(fasta_path), str(prefix), str(td / "tmp"),
               "--min-seq-id", str(min_seq_id), "-c", str(coverage)]
        logger.info("running: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        tsv = Path(f"{prefix}_cluster.tsv")
        mapping = {}
        with open(tsv) as fh:
            for line in fh:
                rep, member = line.rstrip().split("\t")
                mapping[member] = rep
    if out_csv:
        with open(out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["protein_id", "cluster"])
            for k, v in mapping.items():
                w.writerow([k, v])
        logger.info("wrote cluster map (%d seqs, %d clusters) to %s",
                    len(mapping), len(set(mapping.values())), out_csv)
    return mapping
