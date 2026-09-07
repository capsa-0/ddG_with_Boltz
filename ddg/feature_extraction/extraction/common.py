"""
Module: common
Description: Sharding and resumability helpers shared by every backbone runner.

Each backbone (Boltz-2, ESMFold, ...) has its own runner, but they all consume the
same query YAMLs and converge on the same canonical output contract:

    <raw_features_dir>/predictions/<key>/embeddings_<key>.npz

holding `s` (L x Ds), `z` (L x L x Dz) and, when the backbone has one,
`pdistogram` (L x L x bins). `ddg.storage.slim` reads exactly those names, so the
slim / features / evaluation steps are unaffected by which backbone produced them.

Keeping the shard split and the "already done" test here means every backbone gets
identical resume semantics: a requeued array only redoes leftover work, and a node
dying mid-shard costs one structure, not the shard.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def shard_files(all_files, shard):
    """Deterministic round-robin split: shard (i, n) -> files[i::n]."""
    i, n = shard
    if n <= 0 or not (0 <= i < n):
        raise ValueError(f"invalid shard {shard}")
    return all_files[i::n]


def is_done(dst_predictions: Path, key: str) -> bool:
    """True if this query already has a canonical embeddings prediction on disk."""
    d = Path(dst_predictions) / key
    return d.is_dir() and any(d.glob("embeddings_*.npz"))


def slimmed_keys(config) -> set:
    """Structure keys already compacted into the slim store.

    With incremental slim, a shard deletes its raw NPZs right after slimming them,
    so raw-NPZ existence alone would make predict regenerate them on a rerun. Skip
    anything already in a slim shard as well.
    """
    import numpy as np
    slim_dir = Path(config.exp_processed_dir) / "slim"
    done: set = set()
    if slim_dir.exists():
        for f in slim_dir.glob("*.npz"):
            try:
                with np.load(f, allow_pickle=False) as d:
                    done.update(str(k) for k in d["keys"])
            except Exception:
                pass
    return done


def select_pending(config, dst_predictions: Path, shard=None):
    """Resolve the query files this run should actually predict.

    Returns (pending, label). `pending` is the list of query YAMLs that have
    neither a canonical prediction nor a slim entry yet; `label` describes the
    selection for logging. An empty `pending` means there is nothing to do.
    """
    queries_dir = Path(config.queries_dir)
    if not queries_dir.exists():
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

    all_files = sorted(queries_dir.glob("*.yaml"))
    if not all_files:
        raise FileNotFoundError(f"No query YAML files in {queries_dir}")

    if shard is None:
        files, label = all_files, f"all {len(all_files)} queries"
    else:
        i, n = shard
        files = shard_files(all_files, shard)
        label = f"shard {i}/{n} ({len(files)} of {len(all_files)} queries)"
        if not files:
            logger.warning("shard %d/%d is empty; nothing to do", i, n)
            return [], label

    slimmed = slimmed_keys(config)
    pending = [f for f in files
               if not is_done(dst_predictions, f.stem) and f.stem not in slimmed]
    skipped = len(files) - len(pending)
    if skipped:
        logger.info("Skipping %d already-predicted queries in %s", skipped, label)
    return pending, label


def read_query_sequence(query_yaml: Path) -> str:
    """Pull the protein sequence out of a query YAML.

    The queries are written for Boltz (ddg.feature_extraction.model_inputs.
    queries_generator) as::

        sequences:
          - protein: {id: ..., sequence: ..., msa: ...}

    A backbone that takes a bare sequence (ESMFold) reads it back from here rather
    than from the FASTA, so it sees exactly the chain `prepare` validated the
    mutation against.
    """
    import yaml
    with open(query_yaml) as f:
        doc = yaml.safe_load(f)
    for entry in doc.get("sequences", []):
        protein = entry.get("protein")
        if protein and protein.get("sequence"):
            return protein["sequence"]
    raise ValueError(f"no protein sequence in query {query_yaml}")
