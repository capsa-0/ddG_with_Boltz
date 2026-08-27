"""Is the Boltz-vs-Rosetta gap real, or 13-dataset sampling noise?

The headline is a MEDIAN Spearman over 13 MAVE datasets. A median over 13 moves
easily, so the gap needs an interval before it can be claimed. Two levels of
resampling, and they answer different questions:

  * `--unit protein` (default) resamples the 11 PROTEINS with replacement. Proteins
    are the unit of independence here -- a protein's two MAVE datasets share a
    sequence, a structure, an MSA and a Rosetta calculation, so treating 13 datasets
    as 13 independent draws would understate the uncertainty.
  * `--unit dataset` resamples the 13 datasets. Reported for comparison; it is the
    more permissive of the two.

The quantity that matters is the PAIRED difference (Boltz - Rosetta) within each
resample, not the difference of two independently-bootstrapped medians: both arms are
scored on the same datasets with the same rows and the same folds, so resampling them
together cancels the shared variance and gives a much tighter -- and honest -- interval.

Reads layer2_lopo_per_dataset.csv, which already holds one Spearman per
(model, arm, dataset) from score.py.

    conda run -n ddG_with_Boltz python results/15_mave_stability_transfer/bootstrap.py
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_BOOT = 10000
PAIRED_MODELS = ("ddg_only", "ddg_dde", "position_context")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bootstrap CI on the Boltz-Rosetta gap")
    ap.add_argument("--per-dataset", type=Path,
                    default=HERE / "layer2_lopo_per_dataset.csv")
    ap.add_argument("--unit", choices=("protein", "dataset"), default="protein")
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--drop-ubi4", action="store_true",
                    help="exclude the one protein homologous to Tsuboyama")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    d = pd.read_csv(args.per_dataset)
    if args.drop_ubi4:
        d = d[d["protein"] != "UBI4"]
    rng = np.random.default_rng(args.seed)

    units = sorted(d["protein"].unique() if args.unit == "protein"
                   else d["dataset"].unique())
    key = "protein" if args.unit == "protein" else "dataset"
    print(f"resampling {len(units)} {args.unit}s, {args.n_boot} draws"
          + (" (UBI4 dropped)" if args.drop_ubi4 else ""))

    rows = []
    for model in PAIRED_MODELS:
        sub = d[d["model"] == model]
        wide = sub.pivot_table(index="dataset", columns="arm", values="spearman")
        wide = wide.join(sub.groupby("dataset")[key].first().rename("unit"))
        if not {"rosetta", "boltz"}.issubset(wide.columns):
            continue
        by_unit = {u: g for u, g in wide.groupby("unit")}
        obs = wide["boltz"].median() - wide["rosetta"].median()

        diffs = np.empty(args.n_boot)
        for b in range(args.n_boot):
            pick = rng.choice(units, len(units), replace=True)
            s = pd.concat([by_unit[u] for u in pick if u in by_unit])
            diffs[b] = s["boltz"].median() - s["rosetta"].median()
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        p_gt0 = float((diffs > 0).mean())
        rows.append(dict(model=model, unit=args.unit, drop_ubi4=args.drop_ubi4,
                         rosetta=wide["rosetta"].median(),
                         boltz=wide["boltz"].median(), delta=obs,
                         ci_lo=lo, ci_hi=hi, frac_boot_positive=p_gt0,
                         excludes_zero=bool(lo > 0 or hi < 0)))
        verdict = "EXCLUDES 0" if (lo > 0 or hi < 0) else "spans 0"
        print(f"  {model:18} Rosetta {wide['rosetta'].median():+.3f}  "
              f"Boltz {wide['boltz'].median():+.3f}  "
              f"Δ {obs:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  "
              f"P(Δ>0)={p_gt0:.3f}  {verdict}")

    res = pd.DataFrame(rows)
    out = args.out or (HERE / f"bootstrap_{args.unit}"
                       f"{'_noubi4' if args.drop_ubi4 else ''}.csv")
    res.to_csv(out, index=False)
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
