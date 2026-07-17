"""
Module: build_features
Description: The pipeline's ``features`` step. Turns the slim embedding store into
the decided **raw-Δz** feature table (optionally augmented with raw Δs).

For each mutation at 0-based position ``i`` we read the wild-type and mutant slim
slices (see ddg.storage.slim) and emit, from the pair track ``z``:

  - ``zdiag_0..127`` = ``mut_z[i, i] − wt_z[i, i]``            (the diagonal element)
  - ``zpool_0..127`` = mean over residues of ``mut_z[i, :] − wt_z[i, :]``

and, only when ``s`` was kept in the slim store (``slim.keep_s: true``), from the
single track ``s``:

  - ``sdim_0..Ds-1`` = ``mut_s[i] − wt_s[i]``

Metadata columns ``wt_id, mutation, ddg`` are carried through; non-finite values
are set to NaN. The result is written to ``features_summary.parquet`` — the same
filename the rest of the pipeline (and ddg.evaluation) already consumes.

This replaces the old summary-statistics extractor (ddg.exploration). The z
columns here reproduce the previous ad-hoc ``build_rawz.py`` output exactly.
"""

import argparse
import logging

import numpy as np
import pandas as pd

from ddg.config.config_loader import ProjectConfig
from ddg.storage.slim_store import SlimStore

logger = logging.getLogger(__name__)

Z_DIM = 128  # Boltz-2 pair-track (z) feature dimension


def _raw_z_delta(wt: dict, mut: dict, pos: int) -> np.ndarray:
    """Return [zdiag(128), zpool(128)] for one mutation at 0-based `pos`.

    wt["zrow"] is (P, L, Dz) over the WT's kept positions; mut["zrow"] is
    (1, L, Dz) for the mutant's own position. We select the WT row matching
    `pos`, take the diagonal difference at `pos` and the residue-mean of the row
    difference.
    """
    wt_pos = [int(p) for p in wt["pos"]]
    wt_row = wt["zrow"][wt_pos.index(pos)].astype(np.float32)   # (L, Dz)
    mut_row = mut["zrow"][0].astype(np.float32)                 # (L, Dz)
    zdiag = mut_row[pos] - wt_row[pos]                          # (Dz,)
    zpool = (mut_row - wt_row).mean(axis=0)                     # (Dz,)
    return np.concatenate([zdiag, zpool])


def _raw_s_delta(wt: dict, mut: dict, pos: int) -> np.ndarray:
    """Return sdim(Ds) = mut_s[pos] − wt_s[pos] for one mutation.

    Both s tracks are the full single representation (L, Ds); we take the
    mutated residue's row from each.
    """
    wt_s = wt["s"].astype(np.float32)
    mut_s = mut["s"].astype(np.float32)
    return mut_s[pos] - wt_s[pos]


def build_features_frame(config: ProjectConfig) -> pd.DataFrame:
    """Build the raw-Δz [+ Δs] feature table from the slim store."""
    keep_s = bool(config.exp_config.get("slim", {}).get("keep_s", False))
    store = SlimStore(config.exp_processed_dir / "slim")
    mutations = pd.read_csv(config.mutations_df_path)
    logger.info("build_features: %d mutations, keep_s=%s", len(mutations), keep_s)

    meta_rows: list[tuple] = []
    z_feats: list[np.ndarray] = []
    s_feats: list[np.ndarray] = []
    skipped = 0

    for row in mutations.itertuples(index=False):
        pos = int(row.position) - 1
        try:
            wt = store.get(row.wt_key)
            mut = store.get(row.sample_key)
            z = _raw_z_delta(wt, mut, pos)
            s = _raw_s_delta(wt, mut, pos) if keep_s else None
        except Exception as e:  # missing key / position / corrupt slice
            skipped += 1
            logger.debug("build_features: skipping %s %s: %s",
                         row.wt_id, row.mutation, e)
            continue

        meta_rows.append((row.wt_id, row.mutation, float(row.ddg)))
        z_feats.append(z)
        if keep_s:
            s_feats.append(s)

    store.close()

    if not z_feats:
        raise ValueError("build_features: no features produced (empty slim store?)")
    if skipped:
        logger.warning("build_features: skipped %d/%d mutations (missing/corrupt "
                       "slim slices)", skipped, len(mutations))

    df = pd.DataFrame(meta_rows, columns=["wt_id", "mutation", "ddg"])

    Z = np.vstack(z_feats)
    Z = np.where(np.isfinite(Z), Z, np.nan).astype(np.float32)
    z_cols = {f"zdiag_{j}": Z[:, j] for j in range(Z_DIM)}
    z_cols.update({f"zpool_{j}": Z[:, Z_DIM + j] for j in range(Z_DIM)})
    df = pd.concat([df, pd.DataFrame(z_cols)], axis=1)

    if keep_s:
        S = np.vstack(s_feats)
        S = np.where(np.isfinite(S), S, np.nan).astype(np.float32)
        s_cols = {f"sdim_{j}": S[:, j] for j in range(S.shape[1])}
        df = pd.concat([df, pd.DataFrame(s_cols)], axis=1)
        logger.info("build_features: added %d sdim_* columns", S.shape[1])

    return df


def main(experiment_config_path: str,
         names_config_path: str = "ddg/config/internal_config.yaml"):
    """Build features for an experiment and write features_summary.parquet."""
    logger.info("Loading configuration from: %s", experiment_config_path)
    config = ProjectConfig(
        experiment_yaml_path=experiment_config_path,
        internal_yaml_path=names_config_path,
    )
    df = build_features_frame(config)
    out_path = config.exp_processed_dir / "features_summary.parquet"
    df.to_parquet(out_path, index=False)
    logger.info("build_features: wrote %s shape=%s", out_path, df.shape)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description="Build raw-Δz features from slim store")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to experiment YAML configuration")
    args = parser.parse_args()
    main(args.config)
