"""Homology screen for the S669 extension proteins against the training corpus.

results/09 built its MMseqs2 map from the <=500 aa corpus only, so the 9 proteins the
505-701 aa extension adds have never been screened. Worse, the standard check would be
the wrong tool here: `mmseqs easy-cluster` at 80 % coverage cannot flag a short domain
embedded in a long protein, and Tsuboyama's corpus is almost entirely small domains
(median ~70 aa) while every extension protein is 505-701 aa. A 72 aa domain sitting
inside a 648 aa chain would never reach 80 % coverage of the longer sequence.

So this screens with *local* alignment instead, which is coverage-free: Smith-Waterman
of every training sequence against every benchmark sequence, reporting the best local
identity and the length it spans. Run on the original 62 proteins too, as a control --
MMseqs2 called those clean, and a local search should agree.

    python results/16_aftoolkit_headtohead/check_ext_homology.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import Align

ROOT = Path(__file__).resolve().parents[2]
MIN_LEN = 30        # ignore incidental short motifs
FLAG_ID = 0.30      # the identity threshold results/09 used

aligner = Align.PairwiseAligner(scoring="blastp", mode="local")


def best_local(a: str, b: str):
    """Best local alignment: (identity over the aligned block, block length)."""
    try:
        aln = aligner.align(a, b)[0]
    except Exception:
        return 0.0, 0
    sa, sb = aln[0], aln[1]
    pairs = [(x, y) for x, y in zip(sa, sb) if x != "-" and y != "-"]
    if len(pairs) < MIN_LEN:
        return 0.0, len(pairs)
    ident = sum(x == y for x, y in pairs) / len(pairs)
    return ident, len(pairs)


def main():
    tsu = pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/mutations.csv")
    train = tsu.groupby("wt_id").sequence_wt.first()
    print(f"training corpus: {len(train)} proteins, "
          f"median length {int(train.str.len().median())} aa")

    ext = pd.read_csv(ROOT / "data/raw/s669_ext.csv").groupby("uniprot").wt_sequence.first()
    base = pd.read_csv(ROOT / "data/raw/s669.csv").groupby("uniprot").wt_sequence.first()

    rows = []
    for label, targets in (("extension (unscreened)", ext), ("original 62 (control)", base)):
        print(f"\n=== {label}: {len(targets)} proteins ===")
        for pid, seq in targets.items():
            best = (0.0, 0, None)
            for tid, tseq in train.items():
                ident, n = best_local(str(seq), str(tseq))
                if ident > best[0]:
                    best = (ident, n, tid)
            rows.append(dict(group=label, protein=pid, length=len(str(seq)),
                             best_identity=best[0], aligned_len=best[1],
                             closest_training_protein=best[2],
                             leaky=bool(best[0] >= FLAG_ID)))
            if label.startswith("extension"):
                flag = "  <-- FLAG" if best[0] >= FLAG_ID else ""
                print(f"  {pid} ({len(str(seq))} aa): best local identity "
                      f"{best[0]:.1%} over {best[1]} aa vs {best[2]}{flag}")
    df = pd.DataFrame(rows)
    out = Path(__file__).parent / "ext_homology_screen.csv"
    df.to_csv(out, index=False)
    for g, sub in df.groupby("group"):
        print(f"\n{g}: {int(sub.leaky.sum())} of {len(sub)} flagged at >={FLAG_ID:.0%} "
              f"local identity | max identity {sub.best_identity.max():.1%}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
