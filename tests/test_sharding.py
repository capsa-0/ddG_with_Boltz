"""
Tests for predict sharding: the round-robin split must partition all queries
disjointly, and merging shard predictions must land every structure in the
canonical predictions dir. (The actual Boltz call is verified on the cluster.)

Run:  PYTHONPATH=. python tests/test_sharding.py   (or via pytest)
"""

import tempfile
from pathlib import Path

from ddg.feature_extraction.extraction.run_boltz import _shard_files, _merge_predictions
from ddg.cli import _parse_shard


def test_shard_split_partitions_all():
    files = [f"q{i}.yaml" for i in range(10)]
    n = 3
    shards = [_shard_files(files, (i, n)) for i in range(n)]
    # disjoint
    seen = set()
    for s in shards:
        assert not (set(s) & seen)
        seen |= set(s)
    # complete
    assert seen == set(files)
    # balanced-ish
    assert max(len(s) for s in shards) - min(len(s) for s in shards) <= 1
    print("OK: shard split is disjoint, complete, balanced")


def test_parse_shard():
    assert _parse_shard("0/8") == (0, 8)
    assert _parse_shard("3/4") == (3, 4)


def test_merge_predictions():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src = tmp / "boltz_results_x" / "predictions"
        for key in ("p1", "p1_D3Q"):
            d = src / key
            d.mkdir(parents=True)
            (d / f"embeddings_{key}.npz").write_bytes(b"x")
        dst = tmp / "canonical" / "predictions"

        moved = _merge_predictions(src, dst)
        assert moved == 2
        assert (dst / "p1" / "embeddings_p1.npz").exists()
        assert (dst / "p1_D3Q" / "embeddings_p1_D3Q.npz").exists()

        # merging again (already present) must not duplicate or error
        src2 = tmp / "boltz_results_y" / "predictions" / "p1"
        src2.mkdir(parents=True)
        (src2 / "embeddings_p1.npz").write_bytes(b"y")
        moved2 = _merge_predictions(src2.parent, dst)
        assert moved2 == 0
        print("OK: merge collects structures and skips duplicates")


if __name__ == "__main__":
    test_shard_split_partitions_all()
    test_parse_shard()
    test_merge_predictions()
