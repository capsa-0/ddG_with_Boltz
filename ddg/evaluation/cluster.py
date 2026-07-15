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
