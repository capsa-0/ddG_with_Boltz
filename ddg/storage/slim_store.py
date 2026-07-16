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
import pandas as pd
import torch

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


class SlimBoltzDataset:
    """
    Drop-in replacement for BoltzDataset that reads slimmed embeddings.

    Yields the same sample dict shape (wt_id, mut_id, mutation, wt_s/wt_z/...,
    ddg) that ddg.exploration.feature_analysis consumes, so features can be
    computed after the raw NPZs have been deleted.
    """

    def __init__(self, config):
        self.df = pd.read_csv(config.mutations_df_path)
        for col in ("wt_key", "sample_key", "position"):
            if col not in self.df.columns:
                raise ValueError(
                    f"mutations.csv is missing '{col}'; re-run the prepare step "
                    f"so canonical keys are attached before using the slim store."
                )
        self.store = SlimStore(config.exp_processed_dir / "slim")
        logger.debug("SlimBoltzDataset with %d rows", len(self.df))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx) -> dict:
        row = self.df.iloc[idx]
        wt_id, mutation = row["wt_id"], row["mutation"]
        mut_pos = int(row["position"]) - 1

        sample = build_sample(self.store, row["wt_key"], row["sample_key"], mut_pos)
        sample.update({
            "wt_id": wt_id,
            "mut_id": f"{wt_id}_{mutation}",
            "mutation": mutation,
            "ddg": torch.tensor(row["ddg"], dtype=torch.float32),
        })
        return sample
