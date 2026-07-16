"""
Module: slim
Description: Turn raw Boltz embedding NPZs into compact "slim" slices.

Why: Boltz writes s (L x Ds), z (L x L x Dz) and pdistogram (L x L x bins). The
pair tensors scale as L^2 and dominate disk. But the feature code only ever reads
the *row* of the mutated residue from z / pdistogram (z[pos, :] and the diagonal
z[pos, pos]) — never any other row. So we keep:

  - s      : the full single track (L x Ds)          [optional, via keep_s]
  - zrow   : z[positions, :, :]        (P x L x Dz)
  - pdrow  : pdistogram[positions, :]  (P x L x bins)
  - pos    : which residue positions (0-based) the rows correspond to

For a mutant structure P = 1 (its own mutation position). For a wild-type
structure (shared across many mutants) P = every mutated position of that protein.
Values are cast to float16 by default. This is *lossless for the current feature
set* (only the kept row is ever read) and typically ~L/2x smaller on the pair
tensors, plus 2x from float16.

Storage layout: one compressed .npz per shard, holding many structures. Arrays
are named '<field>_<i>' with a parallel 'keys' array mapping index -> structure
key, so no pickling is needed to read them back.
"""

import logging
import shutil
import zipfile
import zlib
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FIELDS = ("s", "zrow", "pdrow", "pos")


def _collapse_single(s: np.ndarray) -> np.ndarray:
    """Drop a leading recycling-step dim if present (s: (steps,L,Ds) -> (L,Ds))."""
    return s[-1] if s.ndim == 3 else s


def _collapse_matrix(m: np.ndarray) -> np.ndarray:
    """Drop a leading recycling-step dim if present (z: (steps,L,L,D) -> (L,L,D))."""
    return m[-1] if m.ndim == 4 else m


def positions_by_structure(mutations_df) -> dict:
    """
    Map each structure key -> sorted list of 0-based residue positions to keep.

    Requires columns wt_key, sample_key, position (1-based) — produced by
    ddg.datasets.prepare.
    """
    out: dict[str, set] = {}
    for row in mutations_df.itertuples(index=False):
        pos0 = int(row.position) - 1
        out.setdefault(row.wt_key, set()).add(pos0)       # WT: all mutated positions
        out.setdefault(row.sample_key, set()).add(pos0)   # mutant: its own position
    return {k: sorted(v) for k, v in out.items()}


def slim_structure(npz_path, positions, keep_s=True, dtype=np.float16) -> dict:
    """Slice one raw embeddings NPZ down to the kept rows."""
    positions = list(positions)
    with np.load(npz_path) as data:
        z = _collapse_matrix(np.squeeze(data["z"]))
        pos = np.asarray(positions, dtype=np.int32)
        out = {
            "pos": pos,
            "zrow": z[pos, :, :].astype(dtype),
        }
        if keep_s:
            s = _collapse_single(np.squeeze(data["s"]))
            out["s"] = s.astype(dtype)
        if "pdistogram" in data:
            pd_ = _collapse_matrix(np.squeeze(data["pdistogram"]))
            out["pdrow"] = pd_[pos, :, :].astype(dtype)
    return out


def write_shard(structure_slims: dict, out_path) -> None:
    """Write many structures' slices into one compressed npz shard."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(structure_slims.keys())
    payload = {"keys": np.asarray(keys)}  # unicode array, no pickle needed
    for i, k in enumerate(keys):
        for field, arr in structure_slims[k].items():
            payload[f"{field}_{i}"] = arr
    np.savez_compressed(out_path, **payload)
    logger.info("Wrote slim shard %s (%d structures)", out_path, len(keys))


def slim_predictions(
    predictions_dir,
    positions_by_struct: dict,
    out_shard,
    keys=None,
    keep_s=True,
    dtype=np.float16,
    delete_raw=False,
) -> list:
    """
    Slim a set of raw prediction folders into one shard.

    Args:
        predictions_dir: dir containing <key>/embeddings_<key>.npz folders.
        positions_by_struct: from positions_by_structure().
        out_shard: output .npz path for this shard.
        keys: structure keys to process (default: all folders present).
        keep_s / dtype: see slim_structure.
        delete_raw: if True, delete each raw folder after slicing it (bounds disk).

    Returns:
        list of structure keys actually written.
    """
    predictions_dir = Path(predictions_dir)
    if keys is None:
        keys = sorted(d.name for d in predictions_dir.iterdir() if d.is_dir())

    slims = {}
    corrupt = []
    for key in keys:
        npz = predictions_dir / key / f"embeddings_{key}.npz"
        if not npz.exists():
            logger.warning("slim: missing embeddings for '%s' (%s)", key, npz)
            continue
        positions = positions_by_struct.get(key)
        if positions is None:
            logger.warning("slim: no known positions for structure '%s'; skipping", key)
            continue
        try:
            slims[key] = slim_structure(npz, positions, keep_s=keep_s, dtype=dtype)
        except (zipfile.BadZipFile, EOFError, zlib.error) as e:
            # Genuine corruption (e.g. a truncated write -> "Bad CRC-32"). The
            # file is worthless, so delete the whole prediction folder: the
            # resumable predict step will regenerate only what is missing. Keep
            # going so every corrupt structure is found in this one pass.
            logger.error("slim: corrupt embeddings for '%s' (%s): %s", key, npz, e)
            shutil.rmtree(predictions_dir / key, ignore_errors=True)
            corrupt.append(key)

    if corrupt:
        raise RuntimeError(
            f"slim: found and deleted {len(corrupt)} corrupt prediction(s): "
            f"{corrupt[:10]}{' ...' if len(corrupt) > 10 else ''}. "
            f"Re-run the predict step (it is resumable and will only redo these), "
            f"then re-run slim."
        )

    write_shard(slims, out_shard)

    if delete_raw:
        for key in slims:
            shutil.rmtree(predictions_dir / key, ignore_errors=True)
        logger.info("slim: deleted %d raw prediction folders", len(slims))

    return list(slims.keys())
