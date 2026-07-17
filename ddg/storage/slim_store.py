"""
Module: slim_store
Description: Read slim embedding shards written by ddg.storage.slim.

SlimStore indexes the per-shard NPZs and returns, for a structure key, the raw
kept slices ``{pos, zrow, [s], [pdrow]}``. The raw-Δz feature builder
(ddg.features.build_features) consumes these slices directly.
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class SlimStore:
    """Index and read slim shards written by ddg.storage.slim.

    Each shard NPZ is opened once and its handle kept open, so per-structure
    reads only seek within the archive instead of re-parsing an ~GB zip central
    directory on every access (that re-open was the dominant cost of the
    features step; keeping the handle open cuts it several-fold).
    """

    def __init__(self, slim_dir):
        self.slim_dir = Path(slim_dir)
        self.files = sorted(self.slim_dir.glob("*.npz"))
        self.index: dict[str, tuple[Path, int]] = {}
        self._handles: dict[Path, "np.lib.npyio.NpzFile"] = {}
        for f in self.files:
            d = np.load(f, allow_pickle=False)   # kept open for the store's life
            self._handles[f] = d
            for i, k in enumerate(d["keys"]):
                self.index[str(k)] = (f, i)
        logger.debug("SlimStore indexed %d structures from %d shard(s)",
                     len(self.index), len(self.files))

    def __contains__(self, key) -> bool:
        return key in self.index

    def get(self, key: str) -> dict:
        """Return the raw slim slice for a structure: {pos, zrow, [s], [pdrow]}."""
        f, i = self.index[key]
        d = self._handles[f]
        out = {"pos": d[f"pos_{i}"], "zrow": d[f"zrow_{i}"]}
        if f"s_{i}" in d:
            out["s"] = d[f"s_{i}"]
        if f"pdrow_{i}" in d:
            out["pdrow"] = d[f"pdrow_{i}"]
        return out

    def close(self) -> None:
        for d in self._handles.values():
            d.close()
        self._handles.clear()
