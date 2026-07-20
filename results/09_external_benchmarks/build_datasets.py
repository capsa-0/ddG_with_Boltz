"""
09_external_benchmarks — build the S669 and Ssym benchmark CSVs from their original
sources into the project's `minimal` schema (uniprot, mutation, wt_sequence, ddg).

Reproducible from scratch (needs network):
    python results/09_external_benchmarks/build_datasets.py

Sources
-------
S669  : DDGemb (Bologna lab — the S669 authors' own canonical UniProt mapping)
        https://ddgemb.biocomp.unibo.it/static/ddgemb/data/S669.{tsv,fasta}
        POS indexes the provided full-length UniProt sequence; all 669 validate
        seq[pos-1]==WT. 19 rows carry a merged '<UniProt>WT<mut>' key -> recovered by regex.
Ssym  : ThermoMPNN repo, data_all/testing/ssym-5fold_clean_dir.csv (carries a SEQ column).
        Positions are PDB-author-numbered -> we solve a UNIQUE constant offset per protein
        and keep only proteins whose offset is unambiguously determined.

Both are capped at <=500 aa (the project's disk-bounded corpus size). Reverse Ssym
mutations are NOT extracted; the reverse prediction is derived analytically from the
antisymmetry-augmented model at scoring time.
"""
import io
import re
import urllib.request
from pathlib import Path

import pandas as pd

CAP = 500
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data/raw"
DDGEMB = "https://ddgemb.biocomp.unibo.it/static/ddgemb/data"
THERMOMPNN = ("https://raw.githubusercontent.com/Kuhlman-Lab/ThermoMPNN/main/"
              "data_all/testing/ssym-5fold_clean_dir.csv")


def _get(url: str) -> str:
    with urllib.request.urlopen(url) as r:
        return r.read().decode()


def build_s669() -> pd.DataFrame:
    seqs, h = {}, None
    for line in _get(f"{DDGEMB}/S669.fasta").splitlines():
        line = line.strip()
        if line.startswith(">"):
            h = line[1:]; seqs[h] = ""
        elif h:
            seqs[h] += line
    tsv = pd.read_csv(io.StringIO(_get(f"{DDGEMB}/S669.tsv")), sep="\t", dtype=str)
    tsv["POS"] = tsv["POS"].astype(int); tsv["DDG"] = tsv["DDG"].astype(float)

    def key(k):  # recover 'O60885WTA420D' -> 'O60885'
        m = re.match(r"^(.+?)WT[A-Z]\d+[A-Z]$", k)
        return m.group(1) if m else k

    rows, bad = [], 0
    for _, r in tsv.iterrows():
        seq = seqs.get(key(r["PDB"]))
        if seq is None:
            bad += 1; continue
        p = r["POS"]
        if 1 <= p <= len(seq) and seq[p - 1] == r["WT"]:
            rows.append({"uniprot": key(r["PDB"]), "mutation": f"{r['WT']}{p}{r['MT']}",
                         "wt_sequence": seq, "ddg": r["DDG"], "seqlen": len(seq)})
        else:
            bad += 1
    assert bad == 0, f"{bad} S669 rows failed position validation"
    return pd.DataFrame(rows)


def build_ssym() -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(_get(THERMOMPNN)))
    rows, dropped = [], []
    for pdb, g in d.groupby("PDB"):
        seq = g["SEQ"].iloc[0]
        fits = [k for k in range(-15, 16)
                if all(0 <= int(m[1:-1]) - 1 + k < len(seq) and seq[int(m[1:-1]) - 1 + k] == m[0]
                       for m in g["MUT"].str.strip())]
        if len(fits) != 1:
            dropped.append((pdb, len(g), fits)); continue
        k = fits[0]
        for _, r in g.iterrows():
            m = r["MUT"].strip()
            rows.append({"uniprot": pdb, "mutation": f"{m[0]}{int(m[1:-1]) + k}{m[-1]}",
                         "wt_sequence": seq, "ddg": float(r["DDG"]), "seqlen": len(seq)})
    if dropped:
        print("Ssym dropped (ambiguous offset):", dropped)
    return pd.DataFrame(rows)


def main():
    for name, df in (("s669", build_s669()), ("ssym", build_ssym())):
        cap = df[df.seqlen <= CAP].drop(columns="seqlen").reset_index(drop=True)
        cap.to_csv(RAW / f"{name}.csv", index=False)
        print(f"{name}: validated={len(df)} -> <=500 kept {len(cap)} variants / "
              f"{cap.uniprot.nunique()} proteins -> {RAW / (name + '.csv')}")


if __name__ == "__main__":
    main()
