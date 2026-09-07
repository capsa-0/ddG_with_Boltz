"""results/17 Phase 1 — run the backbone screen and report the pre-registered endpoint.

Endpoint (pre-registered in status.md before any arm had a number):
pooled **Spearman rho** on FireProt <=200 aa, trained on the 42-protein
Tsuboyama screen subsample, features = TRANSFER_BLOCKS (`zdiag`, 128 d) only,
readout = MLP, trunk frozen. Secondary: per-protein median r, pooled Pearson r.

Fairness is the whole point of this script, so it enforces two things the
individual `ddg.evaluation.transfer` calls cannot:

  * **identical training proteins** — every arm trains on the same 42 proteins.
    The Boltz-2 arm needs no GPU at all: its full 412-protein table is simply
    subset, which is why the incumbent baseline is free.
  * **identical test variants** — the test set is intersected on
    (wt_id, mutation) across every arm before scoring, so no arm is flattered by
    a variant another arm dropped (a structure that OOM'd, say). Without this the
    arms would be scored on different rows and the comparison would be
    uncontrolled.

    python results/17_backbone_transfer/run_screen.py --arms boltz2 esmfold
"""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SCREEN_CSV = ROOT / "data/raw/tsuboyama_screen10.csv"

# arm -> (train feature table, test feature table)
ARMS = {
    "boltz2": ("data/processed/tsuboyama_bench_fast/features_summary.parquet",
               "data/processed/fireprot_le200/features_summary.parquet"),
    "esmfold": ("data/processed/tsuboyama_screen10_esmfold/features_summary.parquet",
                "data/processed/fireprot_le200_esmfold/features_summary.parquet"),
}
KEY = ["wt_id", "mutation"]


def screen_proteins() -> set:
    return {r["protein_id"] for r in csv.DictReader(open(SCREEN_CSV))}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--model", default="mlp", choices=["mlp", "hgb", "svr", "ridge"])
    ap.add_argument("--out", default="data/processed/_analysis/exp17_screen")
    args = ap.parse_args()

    out_root = ROOT / args.out
    out_root.mkdir(parents=True, exist_ok=True)
    prots = screen_proteins()

    # ---- load, and find the test variants every arm actually covers ----
    tables, common = {}, None
    for arm in args.arms:
        tr_p, te_p = (ROOT / p for p in ARMS[arm])
        if not tr_p.exists() or not te_p.exists():
            sys.exit(f"{arm}: missing feature table\n  train {tr_p}\n  test  {te_p}")
        train = pd.read_parquet(tr_p)
        test = pd.read_parquet(te_p)
        train = train[train.wt_id.isin(prots)].reset_index(drop=True)
        tables[arm] = (train, test)
        keys = set(map(tuple, test[KEY].values))
        common = keys if common is None else (common & keys)
        print(f"{arm:8} train {len(train):>5} rows / {train.wt_id.nunique():>3} prot"
              f" | test {len(test):>5} rows / {test.wt_id.nunique():>3} prot")

    print(f"\ntest variants common to all {len(args.arms)} arm(s): {len(common)}")

    # ---- score every arm on the identical intersected test set ----
    rows = []
    for arm in args.arms:
        train, test = tables[arm]
        mask = [tuple(t) in common for t in test[KEY].values]
        test = test[mask].reset_index(drop=True)

        tmp = out_root / arm
        tmp.mkdir(parents=True, exist_ok=True)
        tr_f, te_f = tmp / "train.parquet", tmp / "test.parquet"
        train.to_parquet(tr_f, index=False)
        test.to_parquet(te_f, index=False)

        cmd = [sys.executable, "-m", "ddg.evaluation.transfer",
               "--train", str(tr_f), "--test", str(te_f),
               "--out", str(tmp / "transfer"), "--model", args.model,
               "--label-train", f"Tsuboyama-10%({arm})", "--label-test", "FireProt<=200"]
        print(f"\n=== {arm} ===\n{' '.join(cmd)}")
        subprocess.run(cmd, check=True, cwd=ROOT)

        s = json.loads((tmp / "transfer" / "transfer_summary.json").read_text())
        pp = pd.read_csv(tmp / "transfer" / "per_protein.csv")
        rows.append({
            "arm": arm,
            "n_test": s.get("n"),
            "spearman": s.get("spearman"),
            "pearson": s.get("pearson"),
            "rmse": s.get("rmse"),
            "per_protein_median_r": pp["pearson"].median() if "pearson" in pp else None,
            "sign_flipped": s.get("sign_flipped"),
            "n_train": len(train),
            "n_train_proteins": train.wt_id.nunique(),
        })

    df = pd.DataFrame(rows).sort_values("spearman", ascending=False)
    dest = ROOT / "results/17_backbone_transfer/screen_results.csv"
    df.to_csv(dest, index=False)
    print("\n=== PHASE 1 SCREEN (primary endpoint: pooled Spearman rho) ===")
    print(df.to_string(index=False))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
