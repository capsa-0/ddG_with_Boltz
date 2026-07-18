"""
Build the cross-dataset homology splits for 08_finetune_fireprot.

Pools the WT sequences of Tsuboyama (tsuboyama_bench_fast) and FireProt (fireprot_le200)
and clusters them with MMseqs2 (``mmseqs`` must be on PATH) at each identity threshold
in {0.30, 0.50, 0.90} with 80% coverage (same semantics as the exp-01 homology maps;
coverage-aware set-cover clustering, so no single-linkage chaining). For each threshold
NN in {30,50,90}, whole clusters are assigned to train/test so no train/test pair (within
OR across datasets) shares > threshold identity:
  * MIXED clusters (both Tsu and FP proteins) -> train  (drops the cross-dataset
    homology bridge from every test set),
  * Tsu-only and FP-only clusters -> 80/20 split each (seeded), so both datasets are
    represented in the test side.

Writes to results/08_finetune_fireprot/splits/:
  cluster_map_NN.csv   protein_id,cluster
  split_NN.csv         protein_id,dataset,cluster,set
                       set in {tsu_train, tsu_test, fp_finetune, fp_test}
and prints per-set counts + a leakage assertion. Requires mmseqs on PATH.
"""
import csv
from pathlib import Path

import numpy as np
import pandas as pd

from ddg.evaluation.cluster import _read_fasta, cluster_wt_sequences

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
OUT = ROOT / "results/08_finetune_fireprot/splits"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
TEST_FRAC = 0.20
COVERAGE = 0.8
THRESHOLDS = {30: 0.30, 50: 0.50, 90: 0.90}

tsu = _read_fasta(ROOT / "data/processed/tsuboyama_bench_fast/wt_sequences.fasta")
fp = _read_fasta(ROOT / "data/processed/fireprot_le200/wt_sequences.fasta")
assert not (set(tsu) & set(fp)), "wt_id collision across datasets"
dataset = {**{k: "tsu" for k in tsu}, **{k: "fp" for k in fp}}
combined = OUT / "combined_wt.fasta"
with open(combined, "w") as fh:
    for k, s in {**tsu, **fp}.items():
        fh.write(f">{k}\n{s}\n")
print(f"pooled {len(tsu)+len(fp)} proteins: {len(tsu)} Tsuboyama + {len(fp)} FireProt")

rng = np.random.default_rng(SEED)
for NN, thr in THRESHOLDS.items():
    cmap = cluster_wt_sequences(combined, min_seq_id=thr, coverage=COVERAGE,
                                out_csv=OUT / f"cluster_map_{NN}.csv")
    byc = {}
    for pid, cl in cmap.items():
        byc.setdefault(cl, []).append(pid)
    tsu_clusters, fp_clusters, rows = [], [], []
    for cl, members in byc.items():
        ds = {dataset[m] for m in members}
        kind = "mixed" if ds == {"tsu", "fp"} else next(iter(ds))
        if kind == "tsu":
            tsu_clusters.append(cl)
        elif kind == "fp":
            fp_clusters.append(cl)
        for m in members:
            rows.append([m, dataset[m], cl, kind])

    def pick_test(cluster_list):
        cl = sorted(cluster_list)
        rng.shuffle(cl)
        return set(cl[:max(1, round(len(cl) * TEST_FRAC))])

    tsu_test_cl, fp_test_cl = pick_test(tsu_clusters), pick_test(fp_clusters)

    def assign(pid, ds, cl, kind):
        if kind == "mixed":
            return "tsu_train" if ds == "tsu" else "fp_finetune"
        if ds == "tsu":
            return "tsu_test" if cl in tsu_test_cl else "tsu_train"
        return "fp_test" if cl in fp_test_cl else "fp_finetune"

    split = pd.DataFrame(
        [[pid, ds, cl, assign(pid, ds, cl, kind)] for pid, ds, cl, kind in rows],
        columns=["protein_id", "dataset", "cluster", "set"])
    split.to_csv(OUT / f"split_{NN}.csv", index=False)

    train_cl = set(split[split.set.str.endswith(("train", "finetune"))].cluster)
    test_cl = set(split[split.set.str.endswith("test")].cluster)
    assert not (train_cl & test_cl), f"LEAK at {NN}%: cluster in both train and test"

    n_mixed = sum(1 for m in byc.values() if {dataset[x] for x in m} == {"tsu", "fp"})
    vc = split.set.value_counts().to_dict()
    print(f"\n=== {NN}% identity ({len(byc)} clusters: "
          f"{len(tsu_clusters)} tsu, {len(fp_clusters)} fp, {n_mixed} mixed->train) ===")
    print(f"  tsu_train={vc.get('tsu_train',0)}  tsu_test={vc.get('tsu_test',0)}  "
          f"fp_finetune={vc.get('fp_finetune',0)}  fp_test={vc.get('fp_test',0)}")
print("\nwrote splits to", OUT)
