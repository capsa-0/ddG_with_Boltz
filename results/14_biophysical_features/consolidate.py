"""Merge every per-run result table into results_all.csv, tracking the protocol.

Each run of `run_ablation.py` writes its own `results_<name>.csv`. Those runs differ in
the antisymmetry-augmentation setting, which is an experimental FACTOR here, not a fixed
default — so the merged table carries an explicit `augment` column. (An earlier version
of this file silently mixed protocols across rows, which made two configurations look
comparable when they were not.)

    python results/14_biophysical_features/consolidate.py
"""
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parent

# source table -> (augmentation used?, note)
SRC = {
    "results.csv": (True, "items 1-2, concat-baseline era"),
    "results_bio_t.csv": (True, "item 2 follow-up"),
    "results_cons.csv": (True, "item 3"),
    "results_fp.csv": (True, "first FireProt transfer"),
    "results_fp_cons.csv": (True, "FireProt + conservation"),
    "results_transfer.csv": (True, "S669 predictions persisted"),
    "results_fact_aug.csv": (True, "2x4 factorial"),
    "results_fact_noaug.csv": (False, "2x4 factorial"),
    "results_farctrl.csv": (True, "far-shell control"),
    "results_far_noaug_fp.csv": (False, "far-shell control"),
    "results_headtohead.csv": (False, "vs the prior best (dz)"),
    "results_locality.csv": (False, "locality decomposition"),
    "results_locality_paired.csv": (False, "locality, one paired dump"),
    "results_s669_locality.csv": (False, "S669 replication"),
    "results_s669_base.csv": (False, "S669 concat cell"),
    "results_onehot_s669.csv": (False, "substitution-identity control"),
    "results_onehot_fp.csv": (False, "substitution-identity control"),
}


def main():
    frames = []
    for name, (aug, note) in SRC.items():
        p = OUT / name
        if not p.exists():
            print(f"  (skip, absent) {name}")
            continue
        d = pd.read_csv(p)
        d["augment"], d["source"], d["note"] = aug, name, note
        frames.append(d)
    a = pd.concat(frames, ignore_index=True)
    # a config x set x protocol cell is reproducible; keep the most recent run of it
    a = a.drop_duplicates(subset=["config", "set", "augment"], keep="last")
    a = a.sort_values(["set", "augment", "config"])
    a.to_csv(OUT / "results_all.csv", index=False)
    print(f"\nwrote {OUT / 'results_all.csv'}  ({len(a)} rows)")
    for st in ("tsu_oof", "s669", "fireprot_le500"):
        sub = a[a.set == st]
        if sub.empty:
            continue
        print(f"\n--- {st}: Pearson r ---")
        print(sub.pivot_table(index="config", columns="augment",
                              values="r").round(3).to_string())


if __name__ == "__main__":
    main()
