"""
Module: run_boltz
Description: Execute Boltz embedding prediction on query YAML files, optionally on
one shard of the queries (for parallel SLURM array jobs).

Both the whole-directory and the sharded paths converge on the canonical
predictions directory:  <raw_features_dir>/predictions/<key>/embeddings_<key>.npz
so the slim and features steps are unaffected by how predict was parallelized.
"""

import shutil
import logging
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


def ensure_boltz_cache(config) -> None:
    """Populate the Boltz cache (weights + CCD) once, serially.

    Boltz downloads/extracts its weights and CCD data into --cache on first use.
    When several predict shards start against a *cold* shared cache at once they
    race that download: one writes a partial mols.tar or half-extracted mols/
    dir, another sees it 'exists', skips, and then fails with a tarfile
    ReadError or 'CCD component ... not found'. Warming the cache here (from the
    single, serial prepare step) guarantees the GPU shards only ever read it.
    """
    from boltz.main import download_boltz2

    cache = Path(config.boltz_flags.get("cache", "~/.boltz")).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    logger.info("Warming Boltz cache at %s (weights + CCD)...", cache)
    download_boltz2(cache)
    logger.info("Boltz cache ready at %s", cache)


def _shard_files(all_files, shard):
    """Deterministic round-robin split: shard (i, n) -> files[i::n]."""
    i, n = shard
    if n <= 0 or not (0 <= i < n):
        raise ValueError(f"invalid shard {shard}")
    return all_files[i::n]


def _is_done(dst_predictions: Path, key: str) -> bool:
    """True if this query already has a canonical embeddings prediction on disk."""
    d = dst_predictions / key
    return d.is_dir() and any(d.glob("embeddings_*.npz"))


def _slimmed_keys(config) -> set:
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


def _boltz_cmd(input_path, out_dir, boltz_flags):
    return [
        "boltz", "predict", str(input_path),
        "--out_dir", str(out_dir),
        "--cache", boltz_flags.get("cache", "~/.boltz"),
        "--accelerator", boltz_flags.get("accelerator", "gpu"),
        "--recycling_steps", str(boltz_flags.get("recycling_steps", 3)),
        "--model", boltz_flags.get("model", "boltz2"),
        "--write_embeddings",
        "--embeddings_only",
    ]


def _merge_predictions(src_predictions: Path, dst_predictions: Path) -> int:
    """Move each per-structure prediction folder into the canonical dir."""
    moved = 0
    dst_predictions.mkdir(parents=True, exist_ok=True)
    for child in src_predictions.iterdir():
        if not child.is_dir():
            continue
        target = dst_predictions / child.name
        if target.exists():
            logger.warning("prediction '%s' already present; skipping", child.name)
            continue
        shutil.move(str(child), str(target))
        moved += 1
    return moved


def run_boltz_predictions(config, shard=None) -> None:
    """
    Run Boltz on the query YAMLs (or one shard of them) and collect embeddings.

    Args:
        config: ProjectConfig instance.
        shard: optional (i, n) to process only files[i::n] (SLURM array task).
    """
    queries_dir = Path(config.queries_dir)
    if not queries_dir.exists():
        raise FileNotFoundError(f"Queries directory not found: {queries_dir}")

    all_files = sorted(queries_dir.glob("*.yaml"))
    if not all_files:
        raise FileNotFoundError(f"No query YAML files in {queries_dir}")

    boltz_flags = config.boltz_flags
    dst_predictions = Path(config.raw_features_dir) / "predictions"

    # Select this run's files: the whole directory, or just one shard.
    if shard is None:
        files, tag, label = all_files, "all", f"all {len(all_files)} queries"
    else:
        i, n = shard
        files = _shard_files(all_files, shard)
        tag = f"shard_{i:04d}"
        label = f"shard {i}/{n} ({len(files)} of {len(all_files)} queries)"
        if not files:
            logger.warning("shard %d/%d is empty; nothing to do", i, n)
            return

    # Resumability: skip queries whose canonical prediction already exists, so a
    # resubmitted array only redoes leftover work. A node dying mid-shard then
    # costs at most one structure instead of the whole shard.
    slimmed = _slimmed_keys(config)
    pending = [f for f in files
               if not _is_done(dst_predictions, f.stem) and f.stem not in slimmed]
    skipped = len(files) - len(pending)
    if skipped:
        logger.info("Skipping %d already-predicted queries in %s", skipped, label)
    if not pending:
        logger.info("Nothing to do for %s; all predictions already present", label)
        return
    logger.info("Running Boltz on %s (%d pending)", label, len(pending))

    # Symlink just the pending files into a private input dir (and use a private
    # out dir) so concurrent shards never collide over shared paths.
    base = Path(config.exp_processed_dir) / "_predict_shards"
    input_path = base / f"{tag}_in"
    out_dir = base / f"{tag}_out"
    for d in (input_path, out_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    for f in pending:
        (input_path / f.name).symlink_to(f.resolve())
    tmp_dirs = [input_path, out_dir]

    cmd = _boltz_cmd(input_path, out_dir, boltz_flags)
    logger.info("Executing: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

    # Collect predictions from wherever Boltz wrote them into the canonical dir.
    total = 0
    for pred_dir in Path(out_dir).glob("boltz_results_*/predictions"):
        total += _merge_predictions(pred_dir, dst_predictions)
    for leftover in Path(out_dir).glob("boltz_results_*"):
        shutil.rmtree(leftover, ignore_errors=True)
    for d in tmp_dirs:
        shutil.rmtree(d, ignore_errors=True)

    logger.info("Boltz done: %d prediction folders now under %s", total, dst_predictions)
