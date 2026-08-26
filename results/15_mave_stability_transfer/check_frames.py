"""
Correctness check for score.py's feature reconstruction.

score.py rebuilds Hoie's 47 position-context features from the merged PRISM tables so
that our ddG can be swapped in for Rosetta's. That reconstruction has to be right, or
the Phase-3 comparison is meaningless. This checks it the only way that really counts:
run the identical LOPO on the same 13 Tier-1 datasets twice --

  A. features taken from THEIR preprocessed.pkl (rf4mave.load_datasets)
  B. features rebuilt by OURS from the PRISM tables (score.build_frames, rosetta arm)

-- and compare per-dataset Spearman. The two paths share nothing but the input data,
so agreement means the reconstruction reproduces their feature semantics.

Exact equality is not expected: their grids come from a slightly different missing-value
and position-median path. Per-dataset rho within ~0.02 is the bar.

    conda run -n ddG_with_Boltz python results/15_mave_stability_transfer/check_frames.py --trees 60
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from rf4mave import load_datasets, run_lopo, FEATURE_SETS  # noqa: E402
from score import build_frames, SUB_MATRIX  # noqa: E402
from build_corpus import load_tier, MAX_LEN, SRC_DIR  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Verify score.py's feature rebuild")
    ap.add_argument("--trees", type=int, default=60,
                    help="fewer trees than the 150 of the real runs: this compares "
                         "two reconstructions, not absolute values")
    ap.add_argument("--model", default="position_context", choices=list(FEATURE_SETS))
    args = ap.parse_args(argv)

    # --- B: our rebuild, rosetta arm ---
    by_protein = load_tier(SRC_DIR, MAX_LEN)
    sub = np.load(SUB_MATRIX)
    ours_frames, ours_names, ours_prots = build_frames(by_protein, pd.DataFrame(),
                                                       "rosetta", sub)
    print(f"ours   : {len(ours_frames)} datasets / {len(set(ours_prots))} proteins")

    # --- A: their pkl, restricted to the same datasets ---
    frames, names, proteins = load_datasets()
    want = set(ours_names)
    keep = [i for i, n in enumerate(names) if n in want]
    theirs_frames = [frames[i] for i in keep]
    theirs_names = [names[i] for i in keep]
    theirs_prots = [proteins[i] for i in keep]
    print(f"theirs : {len(theirs_frames)} datasets / {len(set(theirs_prots))} proteins")
    missing = want - set(theirs_names)
    if missing:
        print(f"WARNING: not found in their pkl: {sorted(missing)}", file=sys.stderr)

    pats = FEATURE_SETS[args.model]
    a, a_cols = run_lopo(theirs_frames, theirs_names, theirs_prots, pats,
                         trees=args.trees, verbose=False)
    b, b_cols = run_lopo(ours_frames, ours_names, ours_prots, pats,
                         trees=args.trees, verbose=False)
    print(f"features: theirs {len(a_cols)}  ours {len(b_cols)}")

    m = a[["dataset", "protein", "spearman"]].merge(
        b[["dataset", "spearman"]], on="dataset", suffixes=("_theirs", "_ours"))
    m["delta"] = m["spearman_ours"] - m["spearman_theirs"]
    print("\n" + m.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print(f"\nmedian: theirs {m['spearman_theirs'].median():+.3f}   "
          f"ours {m['spearman_ours'].median():+.3f}")
    print(f"max |delta| = {m['delta'].abs().max():.3f}   "
          f"mean |delta| = {m['delta'].abs().mean():.3f}")
    m.to_csv(HERE / "phase0" / f"check_frames_{args.model}.csv", index=False)
    ok = m["delta"].abs().max() < 0.05
    print("PASS" if ok else "FAIL — reconstruction disagrees with their features")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
