"""
Module: slim_store
Description: Read slim embedding shards and reconstruct the tensors the feature
extractor expects.

The extractor only ever reads the mutation-position row of z / pdistogram, so we
reconstruct a full (L x L x D) tensor that is zero everywhere except that row.
This keeps the existing feature code unchanged while storing only the row on disk.
(The zeros are never read by the extractor, so features are identical; the full
allocation is transient and per-sample. Proteins here are small, so this is fine;
a memory-lean row-native extractor is a possible later optimization.)
"""

import logging
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)


class SlimStore:
    """Index and read slim shards written by ddg.storage.slim."""

    def __init__(self, slim_dir):
        self.slim_dir = Path(slim_dir)
        self.files = sorted(self.slim_dir.glob("*.npz"))
        self.index: dict[str, tuple[Path, int]] = {}
        for f in self.files:
            with np.load(f, allow_pickle=False) as d:
                keys = d["keys"]
            for i, k in enumerate(keys):
                self.index[str(k)] = (f, i)
        logger.debug("SlimStore indexed %d structures from %d shard(s)",
                     len(self.index), len(self.files))

    def __contains__(self, key) -> bool:
        return key in self.index

    def get(self, key: str) -> dict:
        """Return the raw slim slice for a structure: {pos, zrow, [s], [pdrow]}."""
        f, i = self.index[key]
        with np.load(f, allow_pickle=False) as d:
            out = {"pos": d[f"pos_{i}"], "zrow": d[f"zrow_{i}"]}
            if f"s_{i}" in d:
                out["s"] = d[f"s_{i}"]
            if f"pdrow_{i}" in d:
                out["pdrow"] = d[f"pdrow_{i}"]
        return out


def _reconstruct_pair(slim: dict, field: str, mut_pos: int) -> torch.Tensor:
    """Rebuild a full (L, L, D) tensor with only row `mut_pos` filled."""
    rows = slim[field]                      # (P, L, D)
    pos_list = [int(p) for p in slim["pos"]]
    idx = pos_list.index(mut_pos)
    L, D = rows.shape[1], rows.shape[2]
    full = np.zeros((L, L, D), dtype=np.float32)
    full[mut_pos] = rows[idx].astype(np.float32)
    return torch.from_numpy(full)


def build_sample(store: SlimStore, wt_key: str, mut_key: str, mut_pos: int) -> dict:
    """
    Reconstruct a sample dict compatible with
    ddg.exploration.feature_analysis.extractors.extract_features.

    Requires s to have been kept (keep_s=True) for s-based features.
    """
    wt = store.get(wt_key)
    mut = store.get(mut_key)

    sample: dict = {}
    if "s" in wt and "s" in mut:
        sample["wt_s"] = torch.from_numpy(wt["s"].astype(np.float32))
        sample["mut_s"] = torch.from_numpy(mut["s"].astype(np.float32))

    sample["wt_z"] = _reconstruct_pair(wt, "zrow", mut_pos)
    sample["mut_z"] = _reconstruct_pair(mut, "zrow", mut_pos)

    if "pdrow" in wt and "pdrow" in mut:
        sample["wt_pdistogram"] = _reconstruct_pair(wt, "pdrow", mut_pos)
        sample["mut_pdistogram"] = _reconstruct_pair(mut, "pdrow", mut_pos)

    return sample
