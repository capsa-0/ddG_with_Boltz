"""Build the results/17 Phase 1 screen corpus: a 10 % protein subsample of Tsuboyama.

Why a subsample. results/03 measured the learning curve over *proteins*: 33
training proteins already reach pooled r 0.744, 330 reach 0.793, and the
seed-to-seed SD is <= 0.002. The backbone differences this experiment is trying
to resolve are 0.05-0.10 r (theory/sota_2026.md: AF2 0.56 / MSA-Transformer 0.53
/ ESM-2 0.47), an order of magnitude above that noise floor. So a 10 % subsample
can *rank* the arms at a tenth of the GPU cost, and only the survivors pay for
the full corpus.

What it does NOT support: publishable absolute numbers (the 10 % point sits
~0.05 r below the full corpus), and it assumes the arms do not differ in data
efficiency -- an arm that loses at 41 proteins could in principle win at 412.
Both limitations are recorded in status.md.

The subsample is FIXED and SEEDED: every arm must screen on the identical
proteins, or the comparison is not controlled. Stratified by `is_natural` so the
natural/designed mix matches the parent corpus (results/01 holds out `denovo`).

    python results/17_backbone_transfer/build_screen_corpus.py
"""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/tsuboyama_bench_fast.csv"
OUT = ROOT / "data/raw/tsuboyama_screen10.csv"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fraction", type=float, default=0.10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.src)))
    by_protein = defaultdict(list)
    for r in rows:
        by_protein[r["protein_id"]].append(r)

    # stratify on is_natural, which is a per-protein property here
    strata = defaultdict(list)
    for pid, rs in by_protein.items():
        strata[rs[0].get("is_natural", "")].append(pid)

    rng = random.Random(args.seed)
    chosen = []
    for label, pids in sorted(strata.items()):
        pids = sorted(pids)                       # deterministic before shuffling
        rng.shuffle(pids)
        k = max(1, round(len(pids) * args.fraction))
        chosen.extend(pids[:k])
    chosen = sorted(chosen)

    out_rows = [r for pid in chosen for r in by_protein[pid]]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(out_rows)

    print(f"parent : {len(by_protein)} proteins, {len(rows)} mutations  ({args.src})")
    print(f"screen : {len(chosen)} proteins, {len(out_rows)} mutations  "
          f"(fraction={args.fraction}, seed={args.seed})")
    for label, pids in sorted(strata.items()):
        got = sum(1 for p in chosen if p in set(pids))
        print(f"  is_natural={label!r}: {got}/{len(pids)} proteins")
    print(f"structures to predict: {len(out_rows) + len(chosen)}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
