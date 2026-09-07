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

from ddg.feature_extraction.extraction.common import is_done, select_pending

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


def _boltz_cmd(input_path, out_dir, boltz_flags):
    cmd = [
        "boltz", "predict", str(input_path),
        "--out_dir", str(out_dir),
        "--cache", boltz_flags.get("cache", "~/.boltz"),
        "--accelerator", boltz_flags.get("accelerator", "gpu"),
        "--recycling_steps", str(boltz_flags.get("recycling_steps", 3)),
        "--model", boltz_flags.get("model", "boltz2"),
        "--write_embeddings",
        "--embeddings_only",
    ]
    # Opt-in: force Boltz's pure-torch path instead of the optimized triangular-mult
    # kernel. The kernel imports cuequivariance_torch, which is not installed here;
    # Boltz enables it based on GPU arch, so some nodes (e.g. nodo11) crash while
    # others (nodo8) fall back silently. --no_kernels makes predict node-independent.
    if boltz_flags.get("no_kernels"):
        cmd.append("--no_kernels")
    return cmd


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
    boltz_flags = config.backbone_flags
    dst_predictions = Path(config.raw_features_dir) / "predictions"

    pending, label = select_pending(config, dst_predictions, shard=shard)
    if not pending:
        logger.info("Nothing to do for %s; all predictions already present", label)
        return
    tag = "all" if shard is None else f"shard_{shard[0]:04d}"

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

    # Boltz can drop a structure it cannot fit in VRAM and still exit 0: the CLI
    # reports success having written nothing, so `check=True` above never fires and
    # the corpus silently loses that structure. Measured on an 8 GB RTX 2080
    # (compute capability 7.5, so Boltz's own triangle kernels are off): chains up to
    # ~701 aa succeed at ~7.5 GB peak, and >=795 aa are dropped exactly this way.
    # Fail loudly instead -- predict is resumable, so a requeue only redoes the gap.
    missing = [f.stem for f in pending if not is_done(dst_predictions, f.stem)]
    if missing:
        shown = ", ".join(missing[:10]) + (" ..." if len(missing) > 10 else "")
        raise RuntimeError(
            f"Boltz exited 0 but wrote no prediction for {len(missing)} of "
            f"{len(pending)} queries in {label}: {shown}. This is usually a GPU "
            f"out-of-memory on a long chain, which Boltz does not report as an error."
        )
