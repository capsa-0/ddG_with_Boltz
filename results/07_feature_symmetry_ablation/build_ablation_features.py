"""
Build the feature table for 07_feature_symmetry_ablation from a slim embedding store.

Emits, per mutation at 0-based position i, from the Boltz pair track z:
  Δz (difference; the current pipeline's features):
    zdiag_0..127 = mut_z[i,i] - wt_z[i,i]
    zpool_0..127 = mean_j( mut_z[i,j] - wt_z[i,j] )
  concat (the old-notebook features; keeps both absolute levels):
    wtz_0..127   = mean_j( wt_z[i,j] )     (WT pooled row)
    mtz_0..127   = mean_j( mut_z[i,j] )    (mutant pooled row)
Note zpool == mtz - wtz, so Δz is recoverable from concat but not vice-versa.

    python results/07_feature_symmetry_ablation/build_ablation_features.py <processed_dir>
writes <processed_dir>/features_ablation.parquet.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ddg.storage.slim_store import SlimStore

Z = 128


def features(wt, mut, pos):
    wt_pos = [int(p) for p in wt["pos"]]
    wt_row = wt["zrow"][wt_pos.index(pos)].astype(np.float32)   # (L, Dz)
    mut_row = mut["zrow"][0].astype(np.float32)                 # (L, Dz)
    zdiag = mut_row[pos] - wt_row[pos]
    zpool = (mut_row - wt_row).mean(axis=0)
    wtz = wt_row.mean(axis=0)
    mtz = mut_row.mean(axis=0)
    return np.concatenate([zdiag, zpool, wtz, mtz])   # 4*128


def main(proc_dir):
    proc = Path(proc_dir)
    store = SlimStore(proc / "slim")
    mut = pd.read_csv(proc / "mutations.csv")
    meta, feats, skipped = [], [], 0
    for r in mut.itertuples(index=False):
        pos = int(r.position) - 1
        try:
            w = store.get(r.wt_key)
            m = store.get(r.sample_key)
            feats.append(features(w, m, pos))
        except Exception:
            skipped += 1
            continue
        meta.append((r.wt_id, r.mutation, float(r.ddg)))
    store.close()
    F = np.where(np.isfinite(np.vstack(feats)), np.vstack(feats), np.nan).astype(np.float32)
    df = pd.DataFrame(meta, columns=["wt_id", "mutation", "ddg"])
    blocks = ["zdiag", "zpool", "wtz", "mtz"]
    cols = {f"{b}_{j}": F[:, k * Z + j] for k, b in enumerate(blocks) for j in range(Z)}
    df = pd.concat([df, pd.DataFrame(cols)], axis=1)
    out = proc / "features_ablation.parquet"
    df.to_parquet(out, index=False)
    print(f"{proc.name}: wrote {out} shape={df.shape} (skipped {skipped})")


if __name__ == "__main__":
    main(sys.argv[1])
