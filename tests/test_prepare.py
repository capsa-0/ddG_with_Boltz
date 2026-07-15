"""
Tests for ddg.datasets.prepare: mutation parsing + WT-identity validation.

Run:  PYTHONPATH=. python tests/test_prepare.py   (or via pytest)
"""

import pandas as pd

from ddg.datasets.prepare import parse_mutation, prepare_mutations_frame


def test_parse_mutation():
    assert parse_mutation("P8A") == ("P", 8, "A")
    assert parse_mutation("d1q") == ("D", 1, "Q")   # case-insensitive
    assert parse_mutation("A12") is None            # missing mut aa
    assert parse_mutation("") is None
    assert parse_mutation("XxY") is None


def test_validation_and_keys():
    df = pd.DataFrame([
        # valid: seq[0]=='D' matches D1Q
        {"sample_id": "s0", "wt_id": "EA|p.pdb", "mutation": "D1Q",
         "sequence_wt": "DEVTI", "ddg": 0.1},
        # WT mismatch: says A1V but seq[0]=='D'
        {"sample_id": "s1", "wt_id": "EA|p.pdb", "mutation": "A1V",
         "sequence_wt": "DEVTI", "ddg": 0.2},
        # out of range: position 99
        {"sample_id": "s2", "wt_id": "p2", "mutation": "T99A",
         "sequence_wt": "DEVTI", "ddg": 0.3},
        # bad format
        {"sample_id": "s3", "wt_id": "p2", "mutation": "junk",
         "sequence_wt": "DEVTI", "ddg": 0.4},
        # non-standard aa (mutant to X)
        {"sample_id": "s4", "wt_id": "p2", "mutation": "D1X",
         "sequence_wt": "DEVTI", "ddg": 0.5},
    ])
    clean, report = prepare_mutations_frame(df)

    assert report.input_rows == 5
    assert report.output_rows == 1
    assert report.dropped_wt_mismatch == 1
    assert report.dropped_out_of_range == 1
    assert report.dropped_bad_format == 1
    assert report.dropped_nonstandard_aa == 1

    # the surviving row has canonical keys and parsed fields
    r = clean.iloc[0]
    assert r["wt_aa"] == "D" and r["position"] == 1 and r["mut_aa"] == "Q"
    assert r["wt_key"] == "EA_p.pdb"           # '|' sanitized
    assert r["sample_key"] == "EA_p.pdb_D1Q"
    print("OK: validation drops bad rows and attaches canonical keys")


if __name__ == "__main__":
    test_parse_mutation()
    test_validation_and_keys()
