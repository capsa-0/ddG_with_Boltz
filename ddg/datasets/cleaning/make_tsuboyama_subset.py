"""
Build a bounded, family-diverse subset of the Tsuboyama single-mutant dataset for
a runnable experiment.

Selects N proteins (each is a small domain = a holdout group) and up to K valid
single mutants per protein, keeping the natural ddG distribution (so near-neutral
and destabilizing mutations are both represented). Rows whose WT amino acid does
not match the sequence are dropped.

Usage:
    python -m ddg.datasets.cleaning.make_tsuboyama_subset \
        --input data/raw/tsuboyama_single_mutants_ddg.csv \
        --output data/raw/tsuboyama_subset.csv \
        --n-proteins 60 --mutants-per-protein 35 --seed 42
"""

import argparse
import logging

import pandas as pd

from ddg.datasets.prepare import parse_mutation, STANDARD_AA

logger = logging.getLogger(__name__)
OUT_COLUMNS = ["protein_id", "wt_sequence", "mutation", "ddg"]


def _valid_mask(sub: pd.DataFrame) -> pd.Series:
    seq = sub["wt_sequence"].iloc[0]
    def ok(mut):
        parsed = parse_mutation(mut)
        if parsed is None:
            return False
        wt_aa, pos, mut_aa = parsed
        if wt_aa not in STANDARD_AA or mut_aa not in STANDARD_AA:
            return False
        return 1 <= pos <= len(seq) and seq[pos - 1] == wt_aa
    return sub["mutation"].map(ok)


def build_subset(df, n_proteins, mutants_per_protein, seed, min_len, max_len):
    df = df.copy()
    df["L"] = df["wt_sequence"].str.len()

    lens = df.groupby("protein_id")["L"].first()
    eligible = lens[(lens >= min_len) & (lens <= max_len)].index.tolist()
    counts = df[df.protein_id.isin(eligible)].groupby("protein_id").size()
    eligible = counts[counts >= mutants_per_protein].index.tolist()

    rng = pd.Series(eligible).sample(
        n=min(n_proteins, len(eligible)), random_state=seed
    ).tolist()

    parts = []
    for pid in rng:
        sub = df[df.protein_id == pid]
        sub = sub[_valid_mask(sub)]
        if sub.empty:
            continue
        take = sub.sample(n=min(mutants_per_protein, len(sub)), random_state=seed)
        parts.append(take)

    out = pd.concat(parts, ignore_index=True)[OUT_COLUMNS]
    return out


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/tsuboyama_single_mutants_ddg.csv")
    ap.add_argument("--output", required=True)
    ap.add_argument("--n-proteins", type=int, default=60)
    ap.add_argument("--mutants-per-protein", type=int, default=35)
    ap.add_argument("--min-len", type=int, default=32)
    ap.add_argument("--max-len", type=int, default=72)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    out = build_subset(df, args.n_proteins, args.mutants_per_protein,
                       args.seed, args.min_len, args.max_len)
    out.to_csv(args.output, index=False)

    n_prot = out.protein_id.nunique()
    n_struct = len(out) + n_prot
    logger.info(
        "Wrote %s: %d mutants across %d proteins -> %d Boltz structures "
        "(%d WT + %d mutant). Near-neutral |ddg|<0.5: %.0f%%.",
        args.output, len(out), n_prot, n_struct, n_prot, len(out),
        100 * (out.ddg.abs() < 0.5).mean(),
    )


if __name__ == "__main__":
    main()
