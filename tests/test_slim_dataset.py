"""
End-to-end check that features computed from the slim store (after the raw NPZs
are deleted) match features computed from the raw Boltz output.

Run:  PYTHONPATH=. python tests/test_slim_dataset.py   (or via pytest)
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ddg.datasets.boltz_dataset import BoltzDataset
from ddg.storage.slim import positions_by_structure, slim_predictions
from ddg.storage.slim_store import SlimBoltzDataset
from ddg.exploration.feature_analysis.extractors import extract_features

L, DS, DZ, BINS = 10, 5, 4, 6


def _raw(predictions: Path, key: str, seed: int):
    rng = np.random.default_rng(seed)
    d = predictions / key
    d.mkdir(parents=True)
    np.savez(d / f"embeddings_{key}.npz",
             s=rng.standard_normal((L, DS)).astype(np.float32),
             z=rng.standard_normal((L, L, DZ)).astype(np.float32),
             pdistogram=rng.standard_normal((L, L, BINS)).astype(np.float32))


def test_slim_dataset_matches_raw_and_survives_deletion():
    wt_id, mutation = "EA|p.pdb", "D3Q"
    wt_key, sample_key = "EA_p.pdb", "EA_p.pdb_D3Q"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        exp_dir = tmp / "exp"
        raw_dir = exp_dir / "boltz_raw_output"
        predictions = raw_dir / "predictions"
        _raw(predictions, wt_key, 1)
        _raw(predictions, sample_key, 2)

        df = pd.DataFrame([{
            "wt_id": wt_id, "mutation": mutation, "sequence_wt": "A" * L, "ddg": 0.5,
            "wt_key": wt_key, "sample_key": sample_key, "position": 3,
        }])
        mutations_csv = exp_dir / "mutations.csv"
        exp_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(mutations_csv, index=False)

        config = SimpleNamespace(
            mutations_df_path=mutations_csv,
            raw_features_dir=raw_dir,
            exp_processed_dir=exp_dir,
        )

        # raw-path features
        raw_ds = BoltzDataset(config)
        raw_feats = extract_features(raw_ds[0], mut_pos=2)

        # slim (float32 for exactness), then DELETE raw
        slim_predictions(predictions, positions_by_structure(df),
                         exp_dir / "slim" / "shard_0000.npz",
                         keep_s=True, dtype=np.float32, delete_raw=True)
        assert not (predictions / wt_key).exists()   # raw really gone

        slim_ds = SlimBoltzDataset(config)
        slim_feats = extract_features(slim_ds[0], mut_pos=2)

        assert slim_ds[0]["wt_id"] == wt_id
        for k in raw_feats:
            a, b = float(raw_feats[k]), float(slim_feats[k])
            if np.isnan(a) and np.isnan(b):
                continue
            assert abs(a - b) < 1e-4, f"{k}: {a} vs {b}"
        print("OK: slim dataset reproduces raw features after raw NPZs deleted")


if __name__ == "__main__":
    test_slim_dataset_matches_raw_and_survives_deletion()
