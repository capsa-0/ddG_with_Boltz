"""
Leakage check: are any of the 11 Tier-1 MAVE proteins homologous to our training
corpora (Tsuboyama, FireProt)?

Same method and thresholds as results/09_external_benchmarks/build_homology_map.py --
pool the WT sequences with source-tagged headers, cluster with MMseqs2 at 25% and 30%
identity / 80% coverage via ddg.evaluation.cluster.cluster_wt_sequences, and call a MAVE
protein "leaky" when its cluster also contains a training protein.

Watch UBI4 (ubiquitin), SUMO1 and UBE2I (ubiquitin fold) in particular: Tsuboyama's
corpus is dense in small designed and natural domains, so these are where a hit is
plausible on priors rather than by accident.

    MMSEQS_BIN=/path/to/mmseqs \
      conda run -n ddG_with_Boltz python results/15_mave_stability_transfer/build_homology_map.py
"""
import os
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

_bin = os.environ.get("MMSEQS_BIN")
if _bin:
    os.environ["PATH"] = str(Path(_bin).parent) + os.pathsep + os.environ["PATH"]

from ddg.evaluation.cluster import cluster_wt_sequences  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "09_external_benchmarks"))
from build_homology_map import read_fasta, load_training  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "homology"
THRESHOLDS = (0.25, 0.30)
CORPUS = ROOT / "data" / "raw" / "mave_hoie_le200.csv"
LABELS = ROOT / "data" / "raw" / "mave_hoie_le200_labels.csv"


def main() -> int:
    OUT.mkdir(exist_ok=True)
    tsu, fp = load_training()
    print(f"training: Tsuboyama={len(tsu)}  FireProt={len(fp)} proteins")

    corpus = pd.read_csv(CORPUS)
    seqs = corpus.groupby("uniprot")["wt_sequence"].first().to_dict()
    labels = pd.read_csv(LABELS)
    n_scored = labels.groupby("protein").size().to_dict()
    n_sets = labels.groupby("protein")["dataset"].nunique().to_dict()
    print(f"MAVE Tier 1: {len(seqs)} proteins, "
          f"{labels['dataset'].nunique()} datasets, {len(labels):,} scored rows")

    tagged = {f"TSU__{k}": v for k, v in tsu.items()}
    tagged.update({f"FP__{k}": v for k, v in fp.items()})
    tagged.update({f"MAVE__{k}": v for k, v in seqs.items()})
    fasta = OUT / "_pool_mave.fasta"
    with open(fasta, "w") as fh:
        for k, v in tagged.items():
            fh.write(f">{k}\n{v}\n")
    print(f"pooled {len(tagged)} sequences -> clustering")

    rows = {p: {"protein": p, "length": len(seqs[p]), "n_datasets": n_sets[p],
                "n_scored": n_scored[p]} for p in seqs}
    for t in THRESHOLDS:
        pct = int(t * 100)
        mapping = cluster_wt_sequences(fasta, min_seq_id=t, coverage=0.8)
        members = defaultdict(set)
        for member, rep in mapping.items():
            members[rep].add(member)
        for prot in seqs:
            cluster = members[mapping[f"MAVE__{prot}"]]
            has_tsu = any(m.startswith("TSU__") for m in cluster)
            has_fp = any(m.startswith("FP__") for m in cluster)
            rows[prot][f"leaky_tsu_{pct}"] = has_tsu
            rows[prot][f"leaky_fp_{pct}"] = has_fp
            rows[prot][f"leaky_any_{pct}"] = has_tsu or has_fp
            rows[prot][f"cluster_{pct}"] = ";".join(
                sorted(m for m in cluster if not m.startswith("MAVE__")))[:200]
    fasta.unlink(missing_ok=True)

    res = pd.DataFrame(rows.values()).sort_values("protein")
    res.to_csv(OUT / "mave_le200_leakage.csv", index=False)
    show = [c for c in res.columns if not c.startswith("cluster_")]
    print("\n" + res[show].to_string(index=False))
    for t in THRESHOLDS:
        pct = int(t * 100)
        leaky = res[res[f"leaky_any_{pct}"]]
        print(f"\n>={pct}% identity: {len(leaky)}/{len(res)} proteins leaky "
              f"({int(leaky['n_scored'].sum()):,} of {int(res['n_scored'].sum()):,} "
              f"scored rows)")
        for _, r in leaky.iterrows():
            print(f"    {r['protein']:8} <- {r[f'cluster_{pct}']}")
    print(f"\nwrote {OUT / 'mave_le200_leakage.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
