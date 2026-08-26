"""
Build the Boltz corpus and the label table for the Tier-1 MAVE proteins.

Tier 1 = the 11 Høie et al. proteins of <=200 aa, carrying 13 of their 39 MAVE
datasets. The size cap is a compute budget, not a selection: median direct
Spearman(Rosetta ddG, s_exp) is 0.301 over these 13 datasets and 0.301 over all
39, so the cap does not favour or disfavour the stability baseline.

We enumerate the FULL L x 19 saturation scan of each protein rather than only the
measured variants. Two reasons: Hoie's best model ("position-context") needs all 20
substitutions at a position, and the extra structures cost only 3.8% here.

Outputs
    data/raw/mave_hoie_le200.csv         uniprot,mutation,wt_sequence  (minimal adapter)
    data/raw/mave_hoie_le200_labels.csv  one row per (dataset, variant) with
                                         s_exp / Rosetta ddG / GEMME ddE / ss / rsa

The WT-identity check that `ddg.datasets.prepare` performs is duplicated here and
asserted to zero failures, following results/09_external_benchmarks/build_datasets.py:
prepare *drops* mismatched rows silently, so a numbering bug would otherwise show up
only as a quiet shortfall much later.
"""

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ddg.scan.mutations import all_point_mutations  # noqa: E402

SRC_DIR = ROOT / "data" / "raw" / "mave_hoie"
PROTEIN_RE = re.compile(r"[0-9]{3}_([A-Za-z0-9-]{3,8})_")
MAX_LEN = 200

COLUMN_MAP = {
    "score_00": "s_exp",
    "gemme_score_01": "gemme_dde",
    "Rosetta_ddg_score_02": "rosetta_ddg",
    "ss_03": "ss",
    "rsa_03": "rsa",
}


def read_prism(path: Path):
    """Parse a merged PRISM file -> (header dict-ish text, DataFrame)."""
    header = []
    with open(path, errors="replace") as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            header.append(line)
    head = "".join(header)
    seq = re.search(r"sequence\s*:\s*(\S+)", head)
    name = re.search(r"name\s*:\s*(.+)", head)
    uniprot = re.search(r"uniprot\s*:\s*(\S+)", head)
    df = pd.read_csv(path, comment="#", sep=r"\s+", na_values=["NA"])
    meta = {
        "sequence": seq.group(1) if seq else "",
        "name": name.group(1).strip() if name else "",
        "uniprot": uniprot.group(1) if uniprot else "",
    }
    return meta, df


def load_tier(src_dir: Path, max_len: int):
    """Group the merged PRISM files by protein, keeping those <= max_len."""
    by_protein = {}
    for path in sorted(src_dir.glob("prism_merged_*.txt")):
        stem = path.stem.replace("prism_merged_", "")
        m = PROTEIN_RE.search(stem)
        if m is None:
            raise ValueError(f"cannot extract protein name from {stem!r}")
        protein = m.group(1)
        meta, df = read_prism(path)
        if len(meta["sequence"]) > max_len:
            continue
        entry = by_protein.setdefault(protein, {"meta": meta, "datasets": {}})
        if entry["meta"]["sequence"] != meta["sequence"]:
            raise ValueError(
                f"{protein}: datasets disagree on the WT sequence "
                f"({len(entry['meta']['sequence'])} aa vs {len(meta['sequence'])} aa)")
        entry["datasets"][stem] = df
    return by_protein


def build(by_protein: dict):
    """-> (corpus DataFrame, labels DataFrame, per-protein report rows)."""
    corpus, labels, report = [], [], []
    single = re.compile(r"^[A-Z]\d+[A-Z]$")
    for protein, entry in sorted(by_protein.items()):
        seq = entry["meta"]["sequence"]
        muts = all_point_mutations(seq, protein)
        corpus.append(muts)

        n_scored = 0
        for dataset, df in sorted(entry["datasets"].items()):
            d = df[df["variant"].astype(str).str.fullmatch(single, na=False)].copy()
            d = d.rename(columns=COLUMN_MAP)
            d = d[d["s_exp"].notna()]
            # Duplicate prepare's WT check -- must be zero failures.
            pos = d["variant"].str[1:-1].astype(int)
            wt = d["variant"].str[0]
            bad = [(v, seq[p - 1]) for v, p, w in zip(d["variant"], pos, wt)
                   if not (1 <= p <= len(seq)) or seq[p - 1] != w]
            if bad:
                raise AssertionError(
                    f"{protein}/{dataset}: {len(bad)} WT mismatches, e.g. {bad[:3]}")
            d.insert(0, "protein", protein)
            d.insert(0, "dataset", dataset)
            keep = ["dataset", "protein", "variant", "s_exp", "rosetta_ddg",
                    "gemme_dde", "ss", "rsa"]
            labels.append(d[[c for c in keep if c in d.columns]])
            n_scored += len(d)

        report.append(dict(protein=protein, length=len(seq),
                           n_datasets=len(entry["datasets"]),
                           n_mutations=len(muts),
                           n_structures=len(muts) + 1,
                           n_scored_rows=n_scored,
                           datasets=";".join(sorted(entry["datasets"]))))
    return (pd.concat(corpus, ignore_index=True),
            pd.concat(labels, ignore_index=True),
            pd.DataFrame(report))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Tier-1 MAVE Boltz corpus")
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--max-len", type=int, default=MAX_LEN)
    ap.add_argument("--name", default="mave_hoie_le200")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    args = ap.parse_args(argv)

    by_protein = load_tier(args.src, args.max_len)
    corpus, labels, report = build(by_protein)

    corpus_path = args.raw_dir / f"{args.name}.csv"
    labels_path = args.raw_dir / f"{args.name}_labels.csv"
    corpus.to_csv(corpus_path, index=False)
    labels.to_csv(labels_path, index=False)

    print(report.drop(columns=["datasets"]).to_string(index=False))
    print(f"\n{len(report)} proteins, {report['n_datasets'].sum()} MAVE datasets")
    print(f"corpus : {len(corpus):,} mutations "
          f"(+{len(report)} wild-types = {report['n_structures'].sum():,} "
          f"Boltz structures)  -> {corpus_path}")
    print(f"labels : {len(labels):,} (dataset, variant) rows -> {labels_path}")
    print(f"         with Rosetta ddG: {labels['rosetta_ddg'].notna().sum():,}"
          f"   with GEMME ddE: {labels['gemme_dde'].notna().sum():,}")
    print("WT-identity check: 0 mismatches across all datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
