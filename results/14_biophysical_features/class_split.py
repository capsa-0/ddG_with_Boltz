"""14 — Break an ablation's OOF predictions down by mutation class.

Reads an `exp14_oof_*.csv` dump (one column of out-of-fold predictions per feature
configuration) and reports, per configuration:

  * natural vs de novo designed proteins -- the pre-registered read-out for item 3.
    Conservation features can only work where there are homologues; a "gain" on
    designed proteins with no alignment is an artifact, not evolutionary signal.
    (`is_natural` comes from ddg.evaluation.labels, the results/01 definition.)
  * the two class deficits results/12 found survive effect-size normalisation:
    ->Pro and buried Gly, scored as MAE / sd(true) so bigger effects don't just
    look harder.

Burial for the Gly breakdown is joined from features_bio.parquet (`site_cn10`),
which is the same Boltz-distogram contact number results/12 used.

    python results/14_biophysical_features/class_split.py --oof exp14_oof_results.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ddg.evaluation.labels import add_label_columns

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
STAB = -0.5


def scores(y, p):
    return {
        "n": len(y),
        "r": float(np.corrcoef(y, p)[0, 1]) if len(y) > 2 else np.nan,
        "rho": float(spearmanr(y, p).statistic) if len(y) > 2 else np.nan,
        "mae": float(np.abs(p - y).mean()),
        "sd_true": float(np.std(y)),
        "mae_over_sd": float(np.abs(p - y).mean() / np.std(y)) if np.std(y) > 0 else np.nan,
        "bias": float((p - y).mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", required=True, help="file in data/processed/_analysis/")
    args = ap.parse_args()

    oof = pd.read_csv(ROOT / "data/processed/_analysis" / args.oof)
    cfgs = [c for c in oof.columns if c not in ("wt_id", "mutation", "ddg")]

    bio = pd.read_parquet(
        ROOT / "data/processed/tsuboyama_bench_fast/features_bio.parquet",
        columns=["wt_id", "mutation", "site_cn10"])
    df = add_label_columns(oof).merge(
        bio.drop_duplicates(["wt_id", "mutation"]), on=["wt_id", "mutation"],
        how="left", validate="many_to_one")
    df["buried"] = df["site_cn10"] > df["site_cn10"].quantile(2 / 3)

    groups = {
        "all": np.ones(len(df), bool),
        "natural": df.is_natural == 1,
        "de novo": df.is_natural == 0,
        "stabilizing": df.ddg < STAB,
        "->Pro": df.mut_aa == "P",
        "buried Gly": (df.wt_aa == "G") & df.buried,
    }

    rows = []
    for cfg in cfgs:
        for name, mask in groups.items():
            sub = df[mask & df[cfg].notna()]
            if len(sub) < 10:
                continue
            rows.append({"config": cfg, "group": name,
                         **scores(sub.ddg.to_numpy(float), sub[cfg].to_numpy(float))})
    res = pd.DataFrame(rows)

    for name in groups:
        blk = res[res.group == name]
        if blk.empty:
            continue
        print(f"\n--- {name}  (n = {int(blk.n.iloc[0])}, "
              f"sd(true) = {blk.sd_true.iloc[0]:.2f}) ---")
        print(blk[["config", "r", "rho", "mae", "mae_over_sd", "bias"]]
              .round(3).to_string(index=False))

    out = ROOT / "results/14_biophysical_features" / f"class_split_{Path(args.oof).stem}.csv"
    res.to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
