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


def _shard_files(all_files, shard):
    """Deterministic round-robin split: shard (i, n) -> files[i::n]."""
    i, n = shard
    if n <= 0 or not (0 <= i < n):
        raise ValueError(f"invalid shard {shard}")
    return all_files[i::n]


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
    tmp_dirs = []

    if shard is None:
        input_path = queries_dir
        out_dir = Path(config.exp_processed_dir)
        logger.info("Running Boltz on all %d queries", len(all_files))
    else:
        i, n = shard
        files = _shard_files(all_files, shard)
        logger.info("Running Boltz on shard %d/%d (%d of %d queries)",
                    i, n, len(files), len(all_files))
        if not files:
            logger.warning("shard %d/%d is empty; nothing to do", i, n)
            return
        base = Path(config.exp_processed_dir) / "_predict_shards"
        input_path = base / f"shard_{i:04d}_in"
        out_dir = base / f"shard_{i:04d}_out"
        for d in (input_path, out_dir):
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True)
        for f in files:  # symlink the shard's yamls into a private input dir
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
