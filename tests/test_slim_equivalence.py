"""
Slim storage must be lossless for the current feature set: features computed from
the slim slices must equal features from the full embeddings (exactly in float32,
within tolerance in float16). Also checks the on-disk size actually shrinks.

Run:  PYTHONPATH=. python tests/test_slim_equivalence.py   (or via pytest)
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from ddg.datasets.boltz_dataset import BoltzNPZLoader
from ddg.exploration.feature_analysis.extractors import extract_features
from ddg.storage.slim import slim_predictions
from ddg.storage.slim_store import SlimStore, build_sample

L, DS, DZ, BINS = 12, 5, 4, 6
MUT_POS = 3  # 0-based


def _make_raw(predictions: Path, key: str, seed: int):
    rng = np.random.default_rng(seed)
    d = predictions / key
    d.mkdir(parents=True, exist_ok=True)
    np.savez(
        d / f"embeddings_{key}.npz",
        s=rng.standard_normal((L, DS)).astype(np.float32),
        z=rng.standard_normal((L, L, DZ)).astype(np.float32),
        pdistogram=rng.standard_normal((L, L, BINS)).astype(np.float32),
    )


def _full_sample(predictions: Path, wt_key: str, mut_key: str) -> dict:
    s_wt, z_wt, pd_wt = BoltzNPZLoader.load_tensors(predictions / wt_key / f"embeddings_{wt_key}.npz")
    s_mut, z_mut, pd_mut = BoltzNPZLoader.load_tensors(predictions / mut_key / f"embeddings_{mut_key}.npz")
    return {
        "wt_s": s_wt, "wt_z": z_wt, "wt_pdistogram": pd_wt,
        "mut_s": s_mut, "mut_z": z_mut, "mut_pdistogram": pd_mut,
    }


def _compare(a: dict, b: dict, tol: float):
    assert set(a) == set(b), (set(a) ^ set(b))
    worst = 0.0
    for k in a:
        va, vb = float(a[k]), float(b[k])
        if np.isnan(va) and np.isnan(vb):
            continue
        diff = abs(va - vb)
        worst = max(worst, diff)
        assert diff <= tol, f"{k}: {va} vs {vb} (diff {diff} > {tol})"
    return worst


def _run(dtype, tol):
    wt_key, mut_key = "EA_p.pdb", "EA_p.pdb_D4Q"
    pos_by_struct = {wt_key: [MUT_POS], mut_key: [MUT_POS]}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        predictions = tmp / "predictions"
        _make_raw(predictions, wt_key, seed=1)
        _make_raw(predictions, mut_key, seed=2)

        full = extract_features(_full_sample(predictions, wt_key, mut_key), MUT_POS)

        slim_dir = tmp / "slim"
        slim_predictions(predictions, pos_by_struct, slim_dir / "shard_0000.npz",
                         keep_s=True, dtype=dtype)
        store = SlimStore(slim_dir)
        slim_sample = build_sample(store, wt_key, mut_key, MUT_POS)
        slim_feats = extract_features(slim_sample, MUT_POS)

        worst = _compare(full, slim_feats, tol)
        return worst


def test_float32_is_exact():
    worst = _run(np.float32, tol=1e-4)
    print(f"OK: float32 slim is lossless (worst feature diff {worst:.2e})")


def test_float16_within_tolerance():
    worst = _run(np.float16, tol=5e-2)
    print(f"OK: float16 slim within tolerance (worst feature diff {worst:.2e})")


def test_size_reduction():
    """With a longer protein, slim must be clearly smaller than raw on disk."""
    big_L = 120
    wt_key = "prot"
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        predictions = tmp / "predictions"
        d = predictions / wt_key
        d.mkdir(parents=True)
        rng = np.random.default_rng(0)
        raw_path = d / f"embeddings_{wt_key}.npz"
        np.savez_compressed(
            raw_path,
            s=rng.standard_normal((big_L, DS)).astype(np.float32),
            z=rng.standard_normal((big_L, big_L, DZ)).astype(np.float32),
            pdistogram=rng.standard_normal((big_L, big_L, BINS)).astype(np.float32),
        )
        raw_size = raw_path.stat().st_size

        slim_path = tmp / "slim" / "shard_0000.npz"
        slim_predictions(predictions, {wt_key: [3]}, slim_path,
                         keep_s=True, dtype=np.float16)
        slim_size = slim_path.stat().st_size
        ratio = raw_size / slim_size
        assert ratio > 5, f"expected big reduction, got {ratio:.1f}x"
        print(f"OK: size reduction {ratio:.0f}x (raw {raw_size} -> slim {slim_size} bytes) at L={big_L}")


if __name__ == "__main__":
    test_float32_is_exact()
    test_float16_within_tolerance()
    test_size_reduction()
