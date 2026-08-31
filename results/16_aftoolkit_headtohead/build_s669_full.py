"""Merge the S669 corpus with the 505-701 aa extension into one 629-variant corpus.

The pipeline's 500-residue cap left 128 of S669's 669 variants unscored. Measuring the
real ceiling (results/16 status log: an 8 GB RTX 2080 handles ~701 aa and silently drops
>=795 aa) showed the 505-701 aa band was reachable, so `s669_ext` extracted its 88
variants on 9 proteins. This concatenates the two feature tables into `s669_full`, which
`run_ablation.py --transfer s669_full` then scores as one corpus: 541 + 88 = 629/669
(94 % of the published benchmark).

    python results/16_aftoolkit_headtohead/build_s669_full.py
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC = ROOT / "data/processed"
OUT = PROC / "s669_full"
TABLES = ("features_ablation.parquet", "features_bio.parquet", "features_msa.parquet")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name in TABLES:
        parts = []
        for exp in ("s669", "s669_ext"):
            path = PROC / exp / name
            if not path.exists():
                print(f"  {name}: {exp} missing -> skipping this table")
                parts = []
                break
            parts.append(pd.read_parquet(path))
        if not parts:
            continue
        base, ext = parts
        overlap = set(zip(base.wt_id, base.mutation)) & set(zip(ext.wt_id, ext.mutation))
        assert not overlap, f"{name}: extension overlaps the base corpus: {list(overlap)[:5]}"
        assert list(base.columns) == list(ext.columns), f"{name}: column mismatch"
        merged = pd.concat([base, ext], ignore_index=True)
        merged.to_parquet(OUT / name, index=False)
        print(f"  {name}: {len(base)} + {len(ext)} = {len(merged)} rows, "
              f"{merged.wt_id.nunique()} proteins")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
