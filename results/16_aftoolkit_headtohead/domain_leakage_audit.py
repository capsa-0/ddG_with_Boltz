"""Do any benchmark proteins contain a training domain verbatim?

results/09 screened S669 with `mmseqs easy-cluster` at 80 % coverage. That is the wrong
instrument for this corpus pairing: Tsuboyama's Megascale entries are small domains
*excised from real proteins* (median ~70 aa), while S669 proteins run to hundreds of
residues. A 54 aa domain sitting verbatim inside a 455 aa benchmark protein can never
reach 80 % coverage of the longer sequence, so clustering reports the pair as unrelated.

This screens the way the corpus pairing demands -- local, coverage-free -- and then asks
the question that decides whether it matters: **does the mutated position fall inside the
shared region?** A shared domain the mutation sits outside of is weak exposure; a shared
domain containing the mutated site means the regressor was trained on that exact site's
stability, in that exact sequence context.

    python results/16_aftoolkit_headtohead/domain_leakage_audit.py
"""
from pathlib import Path

import pandas as pd
from Bio import Align

ROOT = Path(__file__).resolve().parents[2]
MIN_ID, MIN_LEN = 0.90, 40      # a verbatim-or-near domain, not a chance patch
aligner = Align.PairwiseAligner(scoring="blastp", mode="local")


def hits(target: str, train: pd.Series):
    """Training domains matching `target` locally at >=MIN_ID over >=MIN_LEN residues."""
    out = []
    for tid, tseq in train.items():
        try:
            aln = aligner.align(target, str(tseq))[0]
        except Exception:
            continue
        pairs = [(a, b) for a, b in zip(aln[0], aln[1]) if a != "-" and b != "-"]
        if len(pairs) < MIN_LEN:
            continue
        ident = sum(a == b for a, b in pairs) / len(pairs)
        if ident >= MIN_ID:
            (s, e) = aln.aligned[0][0][0], aln.aligned[0][-1][-1]   # span in the target
            out.append((tid, ident, len(pairs), int(s), int(e)))
    return out


def main():
    tsu = pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/mutations.csv")
    train = tsu.groupby("wt_id").sequence_wt.first()

    import sys
    fp = pd.concat([pd.read_csv(ROOT / f"data/processed/fireprot_{d}/mutations.csv")
                    for d in ("le200", "201to500")], ignore_index=True)
    fp = fp.rename(columns={"wt_id": "uniprot", "sequence_wt": "wt_sequence"})
    corpora = {
        "s669 (base 541)": pd.read_csv(ROOT / "data/raw/s669.csv"),
        "s669_ext (added 88)": pd.read_csv(ROOT / "data/raw/s669_ext.csv"),
        "fireprot <=500": fp,
    }
    if len(sys.argv) > 1:
        corpora = {k: v for k, v in corpora.items() if sys.argv[1] in k}
    rows = []
    for label, df in corpora.items():
        seqs = df.groupby("uniprot").wt_sequence.first()
        print(f"\n=== {label}: {len(seqs)} proteins ===")
        for pid, seq in seqs.items():
            for tid, ident, n, s, e in hits(str(seq), train):
                var = df[df.uniprot == pid].copy()
                var["pos"] = var.mutation.str[1:-1].astype(int)
                inside = var[(var.pos > s) & (var.pos <= e)]
                rows.append(dict(corpus=label, protein=pid, prot_len=len(str(seq)),
                                 training_domain=tid, identity=ident, aligned=n,
                                 span_start=s + 1, span_end=e,
                                 variants_total=len(var), variants_in_span=len(inside)))
                print(f"  {pid} ({len(str(seq))} aa) contains {tid} "
                      f"at {ident:.0%} over {n} aa (residues {s+1}-{e}); "
                      f"{len(inside)}/{len(var)} of its variants sit INSIDE that span")
    out = pd.DataFrame(rows)
    import sys
    tag = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""
    p = Path(__file__).parent / f"domain_leakage{tag}.csv"
    out.to_csv(p, index=False)
    if len(out):
        print(f"\nproteins containing a training domain: {out.protein.nunique()}")
        print(f"variants inside a shared domain: {out.variants_in_span.sum()}")
    else:
        print("\nno verbatim training domains found in either corpus")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
