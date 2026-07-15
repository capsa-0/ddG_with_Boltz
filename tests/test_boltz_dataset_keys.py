"""
Regression test for bug 1.1: BoltzDataset must resolve embeddings for ids that
contain characters sanitized in filenames (e.g. Tsuboyama 'EA|run2_...pdb').

Run:  python tests/test_boltz_dataset_keys.py   (or via pytest)
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from ddg.datasets.boltz_dataset import BoltzDataset
from ddg.datasets.ids import wt_key, mutant_key


def _write_fake_embeddings(folder: Path, key: str, L=5, ds=3, dz=4, bins=6):
    """Create predictions/<key>/embeddings_<key>.npz with the expected arrays."""
    d = folder / key
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"embeddings_{key}.npz",
        s=np.random.randn(L, ds).astype(np.float32),
        z=np.random.randn(L, L, dz).astype(np.float32),
        pdistogram=np.random.randn(L, L, bins).astype(np.float32),
    )


def test_lookup_with_special_char_id():
    wt_id = "EA|run2_0325_0005.pdb"   # the pipe is what broke lookups
    mutation = "D1Q"
    seq = "DEVTI"

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # mutations.csv with the RAW id (as prepare would write it)
        df = pd.DataFrame([{
            "wt_id": wt_id, "mutation": mutation,
            "sequence_wt": seq, "ddg": 0.07,
        }])
        mutations_csv = tmp / "mutations.csv"
        df.to_csv(mutations_csv, index=False)

        # Prediction folders are named by the SANITIZED keys (as Boltz writes them)
        predictions = tmp / "raw" / "predictions"
        _write_fake_embeddings(predictions, wt_key(wt_id))
        _write_fake_embeddings(predictions, mutant_key(wt_id, mutation))

        config = SimpleNamespace(
            mutations_df_path=mutations_csv,
            raw_features_dir=tmp / "raw",
        )

        ds = BoltzDataset(config)
        sample = ds[0]  # would raise FileNotFoundError before the fix

        assert sample["wt_id"] == wt_id
        assert sample["mutation"] == mutation
        assert sample["wt_s"].shape[0] == 5
        assert sample["mut_z"].ndim == 3
        assert abs(float(sample["ddg"]) - 0.07) < 1e-6
    print("OK: BoltzDataset resolves embeddings for ids containing '|'")


if __name__ == "__main__":
    test_lookup_with_special_char_id()
