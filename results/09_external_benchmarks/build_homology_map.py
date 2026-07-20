"""
09_external_benchmarks — leakage / homology map (MMseqs2).

Pools the WT sequences of the two training corpora (Tsuboyama, FireProt) with each
benchmark (S669, Ssym) and clusters them with MMseqs2 at 25% and 30% identity / 80%
coverage, via the project's ddg.evaluation.cluster.cluster_wt_sequences (same
easy-cluster semantics used in experiments 07/08). A benchmark protein is "leaky"
w.r.t. a corpus when it lands in a cluster containing a protein from that corpus — the
homology-filtered scoring drops leaky proteins. This mirrors ThermoMPNN's removal of
Megascale homologues (>25% id) from its test benchmarks.

    MMSEQS_BIN=/path/to/mmseqs \
      conda run -n ddG_with_Boltz python results/09_external_benchmarks/build_homology_map.py

(If mmseqs is on PATH already, MMSEQS_BIN is unnecessary.) Writes
results/09_external_benchmarks/homology/{benchmark}_leakage.csv and prints a summary.
"""
import os
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root for `import ddg`

# Make a custom mmseqs discoverable to shutil.which before importing cluster.
_bin = os.environ.get("MMSEQS_BIN")
if _bin:
    os.environ["PATH"] = str(Path(_bin).parent) + os.pathsep + os.environ["PATH"]

from ddg.evaluation.cluster import cluster_wt_sequences  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "homology"
OUT.mkdir(exist_ok=True)
THRESHOLDS = (0.25, 0.30)


def read_fasta(path) -> dict:
    seqs, name, buf = {}, None, []
    for line in open(path):
        line = line.rstrip()
        if line.startswith(">"):
            if name is not None:
                seqs[name] = "".join(buf)
            name, buf = line[1:].strip().split()[0], []
        else:
            buf.append(line)
    if name is not None:
        seqs[name] = "".join(buf)
    return seqs


def load_training():
    tsu = read_fasta(ROOT / "data/processed/tsuboyama_bench_fast/wt_sequences.fasta")
    fp = {}
    for d in ("fireprot_le200", "fireprot_201to500"):
        fp.update(read_fasta(ROOT / "data/processed" / d / "wt_sequences.fasta"))
    return tsu, fp


def benchmark_proteins(csv_name):
    df = pd.read_csv(ROOT / "data/raw" / csv_name)
    seq = df.groupby("uniprot")["wt_sequence"].first().to_dict()
    n = df.groupby("uniprot")["ddg"].size().to_dict()
    return seq, n


def main():
    tsu, fp = load_training()
    print(f"training: Tsuboyama={len(tsu)}  FireProt={len(fp)} proteins")

    for bench, csv_name in (("s669", "s669.csv"), ("ssym", "ssym.csv")):
        bseq, bn = benchmark_proteins(csv_name)
        # Pooled FASTA with source-tagged headers (TSU__/FP__/BEN__) to avoid id clashes.
        tagged = {f"TSU__{k}": v for k, v in tsu.items()}
        tagged.update({f"FP__{k}": v for k, v in fp.items()})
        tagged.update({f"BEN__{k}": v for k, v in bseq.items()})
        fasta = OUT / f"_pool_{bench}.fasta"
        with open(fasta, "w") as fh:
            for k, v in tagged.items():
                fh.write(f">{k}\n{v}\n")

        rows = {p: {"protein": p, "n_variants": bn[p]} for p in bseq}
        for t in THRESHOLDS:
            p = int(t * 100)
            mapping = cluster_wt_sequences(fasta, min_seq_id=t, coverage=0.8)
            # cluster representative -> set of member sources
            from collections import defaultdict
            members = defaultdict(set)
            for member, rep in mapping.items():
                members[rep].add(member)
            for prot in bseq:
                key = f"BEN__{prot}"
                cluster = members[mapping[key]]
                has_tsu = any(m.startswith("TSU__") for m in cluster)
                has_fp = any(m.startswith("FP__") for m in cluster)
                rows[prot][f"leaky_tsu_{p}"] = has_tsu
                rows[prot][f"leaky_fp_{p}"] = has_fp
                rows[prot][f"leaky_any_{p}"] = has_tsu or has_fp
        fasta.unlink(missing_ok=True)

        res = pd.DataFrame(rows.values())
        res.to_csv(OUT / f"{bench}_leakage.csv", index=False)
        nv = int(res["n_variants"].sum())
        print(f"\n=== {bench}: {len(res)} proteins / {nv} variants ===")
        for t in THRESHOLDS:
            p = int(t * 100)
            for src in ("tsu", "fp", "any"):
                lk = res[res[f"leaky_{src}_{p}"]]
                v = int(lk["n_variants"].sum())
                print(f"  >={p}% vs {src:3s}: leaky proteins={len(lk):3d} variants={v:4d}"
                      f"  -> clean variants={nv - v} / {len(res) - len(lk)} proteins")
        print(f"  wrote {OUT / (bench + '_leakage.csv')}")


if __name__ == "__main__":
    main()
