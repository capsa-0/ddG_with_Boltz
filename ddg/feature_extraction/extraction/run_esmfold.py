"""
Module: run_esmfold
Description: ESMFold backbone — extract the folding trunk's single/pair tracks.

ESMFold is the single-sequence arm of results/17. It needs **no upstream patch**:
HuggingFace's `EsmForProteinFolding` already returns the trunk tensors as
first-class output fields (`transformers/models/esm/modeling_esmfold.py`, where
`EsmForProteinFoldingOutput(**structure)` is built from a dict carrying `s_s`,
`s_z` and `distogram_logits`). So this runner is a thin loop that maps them onto
the project's NPZ contract:

    s_s              (L x 1024)      -> "s"
    s_z              (L x L x 128)   -> "z"
    distogram_logits (L x L x 64)    -> "pdistogram"

The pair track is 128-wide, the same as Boltz-2's, so `zdiag` stays 128-d and the
feature builder, TRANSFER_BLOCKS and the readouts run unmodified. The single track
is 1024 rather than 384 — only `slim.keep_s` sees that, and results/14 found the
s-derived block is not what carries transfer, so `keep_s: false` is the sane
default for this arm.

No MSA is read. That is the point of the arm (results/04 measured the MSA
contribution as an *input* ablation; this measures it as an architectural one), and
it is also why this backbone is cheap: no MMseqs2, no a3m parsing, one forward pass.

Cost note: ESMFold's forward also runs the structure module (frames, positions,
pLDDT, pTM) that we throw away. Skipping it would need an upstream patch; it is a
small fraction of the cost next to the ESM-2 3B tower plus the 48-block trunk, so
it is not worth the patch.

Memory: the cluster's cards are RTX 2080 8 GB, compute capability 7.5 (Turing — no
bf16). The language-model tower is therefore run in **fp16** and the trunk is
chunked; see `esmfold_flags` in the experiment YAML.
"""

import gc
import logging
from pathlib import Path

import numpy as np

from ddg.feature_extraction.extraction.common import (read_query_sequence,
                                                      select_pending)

logger = logging.getLogger(__name__)

MODEL_ID = "facebook/esmfold_v1"


def ensure_esmfold_cache(config) -> None:
    """Download the ESMFold weights once, serially, before the predict shards run.

    Same rationale as `ensure_boltz_cache`: parallel shards starting against a cold
    HuggingFace cache race the first download and can leave a half-written blob.
    Called from the (serial) prepare step.

    `snapshot_download` fetches the 8.4 GB checkpoint without building the model,
    so this runs inside the CPU step's modest memory budget -- instantiating it
    here would allocate the whole fp32 model (~8.4 GB of tensors) for nothing.
    """
    from huggingface_hub import snapshot_download

    logger.info("Warming ESMFold cache (%s, ~8.4 GB)...", MODEL_ID)
    path = snapshot_download(MODEL_ID)
    logger.info("ESMFold cache ready at %s", path)


def _load_model(flags):
    """Load ESMFold and apply the low-VRAM settings this cluster needs."""
    import torch
    from transformers import AutoTokenizer, EsmForProteinFolding

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = EsmForProteinFolding.from_pretrained(MODEL_ID, low_cpu_mem_usage=True)
    model.eval()

    device = flags.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # The ESM-2 3B tower dominates VRAM and is inference-only here. fp16 halves it
    # and is safe on Turing (cc 7.5 has fp16 tensor cores but no bf16). The trunk
    # itself stays fp32 -- it feeds the features we actually keep.
    if flags.get("half_esm", True) and device != "cpu":
        model.esm = model.esm.half()
    # Escape hatch for the 8 GB cards. The fp32 checkpoint is 8.4 GB; halving the
    # language tower brings the resident weights to roughly 5 GB, leaving ~3 GB of
    # headroom for activations. If a chain still OOMs after dropping chunk_size,
    # halve the trunk too -- but record it, because the trunk is what produces the
    # features and fp16 there is a change to the measurement, not just to memory.
    if flags.get("half_trunk", False) and device != "cpu":
        logger.warning("half_trunk: running the folding trunk in fp16 -- the "
                       "embeddings this produces are NOT bit-comparable with an "
                       "fp32-trunk arm; note it in the result log")
        model.trunk = model.trunk.half()
    chunk = flags.get("chunk_size", 64)
    if chunk:
        model.trunk.set_chunk_size(int(chunk))
    if device != "cpu":
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info("ESMFold on %s (half_esm=%s, half_trunk=%s, chunk_size=%s)",
                device, flags.get("half_esm", True),
                flags.get("half_trunk", False), chunk)
    return tokenizer, model, device


def _embed_one(tokenizer, model, device, sequence, num_recycles):
    """One forward pass -> the three arrays of the NPZ contract."""
    import torch

    inputs = tokenizer([sequence], return_tensors="pt", add_special_tokens=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs, num_recycles=num_recycles)

    # Batch of one; drop it. float32 on the way out so the NPZ matches what the
    # Boltz path writes and `slim` can cast on its own terms.
    return {
        "s": out.s_s[0].float().cpu().numpy(),
        "z": out.s_z[0].float().cpu().numpy(),
        "pdistogram": out.distogram_logits[0].float().cpu().numpy(),
    }


def _write_prediction(dst_predictions: Path, key: str, arrays: dict) -> None:
    """Write one structure's embeddings at the canonical path, atomically.

    Written to a temporary name and renamed, so a job killed mid-write never
    leaves a truncated NPZ that `is_done` would count as finished.
    """
    out_dir = dst_predictions / key
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / f"embeddings_{key}.npz"
    # np.savez appends '.npz' unless the name already ends in it, so the temp name
    # must too or the rename below has nothing to rename. The leading dot keeps a
    # leftover temp invisible to `is_done`, which globs 'embeddings_*.npz'.
    tmp = out_dir / f".embeddings_{key}.partial.npz"
    np.savez(tmp, **arrays)
    tmp.replace(final)


def run_esmfold_predictions(config, shard=None) -> None:
    """
    Run ESMFold on the query YAMLs (or one shard of them) and write embeddings.

    Args:
        config: ProjectConfig instance.
        shard: optional (i, n) to process only files[i::n] (SLURM array task).
    """
    dst_predictions = Path(config.raw_features_dir) / "predictions"
    pending, label = select_pending(config, dst_predictions, shard=shard)
    if not pending:
        logger.info("Nothing to do for %s; all predictions already present", label)
        return

    flags = config.backbone_flags
    num_recycles = flags.get("recycling_steps")
    logger.info("Running ESMFold on %s (%d pending)", label, len(pending))

    tokenizer, model, device = _load_model(flags)

    failures: list[tuple[str, str]] = []
    for n, query in enumerate(pending, 1):
        key = query.stem
        try:
            sequence = read_query_sequence(query)
            arrays = _embed_one(tokenizer, model, device, sequence, num_recycles)
            _write_prediction(dst_predictions, key, arrays)
        except Exception as e:                      # noqa: BLE001 - reported below
            # One structure that will not fit (or a malformed query) must not cost
            # the whole shard: record it, free the allocator, keep going, and fail
            # loudly at the end so a requeue only redoes the gap.
            failures.append((key, f"{type(e).__name__}: {e}"))
            logger.warning("ESMFold failed on %s: %s", key, e)
            _empty_cache(device)
        if n % 50 == 0:
            logger.info("  %d/%d", n, len(pending))

    _empty_cache(device)
    written = sum(1 for q in pending if (dst_predictions / q.stem).is_dir())
    logger.info("ESMFold done: %d/%d predictions under %s",
                written, len(pending), dst_predictions)

    if failures:
        shown = ", ".join(f"{k} ({e})" for k, e in failures[:10])
        raise RuntimeError(
            f"ESMFold failed on {len(failures)} of {len(pending)} queries in "
            f"{label}: {shown}{' ...' if len(failures) > 10 else ''}. On this "
            f"cluster's 8 GB cards a long chain is the usual cause; predict is "
            f"resumable, so requeueing only redoes the gap."
        )


def _empty_cache(device) -> None:
    import torch
    gc.collect()
    if device != "cpu" and torch.cuda.is_available():
        torch.cuda.empty_cache()
