"""
Compare the Boltz-2 embedding ΔΔG scan against the FoldX scan of the same protein.

Both tables are keyed by mutation string in UniProt numbering (P06280, 32-429), so
they join 1:1 on `mutation`. Only positions present in BOTH are compared -- the
Boltz run is a targeted subset (the 10 positions flagged as overestimated + all
glycines), not a full scan.

Metric choice: **Spearman is the headline**. FoldX values here run to +70 kcal/mol,
which is steric-clash artifact rather than a calibrated ΔΔG, so RMSE/Pearson against
the raw values are dominated by a handful of extreme points. Pearson is reported on
a clipped copy for reference, clearly labelled.

    python results/10_gla_scan/compare_foldx.py \
        --scan data/processed/scan_GLA_human_hard/scan/scan_predictions.csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
FOLDX = HERE / "ddg_varmed_by_mutation_foldx.csv"
# Positions the model was reported to overestimate at (UniProt numbering).
FLAGGED = [80, 137, 169, 200, 201, 228, 301, 325, 360, 409]
CLIP = 10.0   # kcal/mol, for the reference Pearson only


def _corr(x, y):
    """Spearman + Pearson, NaN-safe and safe on degenerate input."""
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3 or np.ptp(x[m]) == 0 or np.ptp(y[m]) == 0:
        return dict(n=int(m.sum()), spearman=np.nan, pearson=np.nan)
    return dict(n=int(m.sum()),
                spearman=float(spearmanr(x[m], y[m]).statistic),
                pearson=float(pearsonr(x[m], y[m])[0]))


def load(scan_path, regime):
    scan = pd.read_csv(scan_path)
    col = f"ddg_{regime}" if regime != "mean" else "ddg_mean"
    if col not in scan.columns:
        raise SystemExit(f"{col} not in {scan_path}; have "
                         f"{[c for c in scan.columns if c.startswith('ddg_')]}")
    fx = pd.read_csv(FOLDX)
    fx = fx[fx.mutation.str[0] != fx.mutation.str[-1]]        # drop X->X rows
    fx = fx.rename(columns={"ddg": "foldx"})[["mutation", "foldx"]]

    df = scan.merge(fx, on="mutation", how="inner")
    df = df.rename(columns={col: "boltz"})
    df["foldx_clipped"] = df.foldx.clip(-CLIP, CLIP)
    return scan, fx, df


def main():
    ap = argparse.ArgumentParser(description="Boltz scan vs FoldX")
    ap.add_argument("--scan", required=True, help="scan_predictions.csv")
    ap.add_argument("--regime", default="mean",
                    help="mean (default) | A_tsuboyama | B_fireprot | D_finetuned")
    ap.add_argument("--out", help="output dir (default: this folder)")
    args = ap.parse_args()
    out = Path(args.out) if args.out else HERE

    scan, fx, df = load(args.scan, args.regime)
    scored = df.dropna(subset=["foldx", "boltz"])
    print(f"scan rows {len(scan)} | FoldX substitutions {len(fx)} | "
          f"joined {len(df)} | both scored {len(scored)}")
    if len(df) < len(scan):
        print(f"  note: {len(scan)-len(df)} scan mutations absent from FoldX")
    missing = df.foldx.isna().sum()
    if missing:
        print(f"  note: {missing} joined rows have no FoldX value "
              f"(FoldX gaps: E58D, V137R, all of L428/L429)")

    rows = []
    overall = _corr(scored.boltz.to_numpy(), scored.foldx.to_numpy())
    clipped = _corr(scored.boltz.to_numpy(), scored.foldx_clipped.to_numpy())
    print(f"\n=== overall (regime {args.regime}) ===")
    print(f"  n={overall['n']}  Spearman={overall['spearman']:+.3f}  "
          f"Pearson(raw)={overall['pearson']:+.3f}  "
          f"Pearson(clip±{CLIP:g})={clipped['pearson']:+.3f}")
    rows.append(dict(group="ALL", **overall, pearson_clipped=clipped["pearson"]))

    # --- glycine vs non-glycine positions ---
    print("\n=== by wild-type residue class ===")
    for label, sel in (("glycine sites (WT=G)", scored.wt_aa == "G"),
                       ("non-glycine sites", scored.wt_aa != "G")):
        if sel.sum() == 0:
            continue
        g = scored[sel]
        c = _corr(g.boltz.to_numpy(), g.foldx.to_numpy())
        print(f"  {label:<22} n={c['n']:<5} Spearman={c['spearman']:+.3f}  "
              f"mean Boltz={g.boltz.mean():+.2f}  mean FoldX={g.foldx.mean():+.2f}  "
              f"bias={g.boltz.mean()-g.foldx.mean():+.2f}")
        rows.append(dict(group=label, **c, pearson_clipped=np.nan,
                         mean_boltz=g.boltz.mean(), mean_foldx=g.foldx.mean()))

    # --- per position, flagged sites called out ---
    print("\n=== per position (flagged sites marked *) ===")
    print(f"  {'pos':>4} {'wt':>2} {'n':>3} {'rho':>7} {'Boltz':>7} {'FoldX':>7} {'bias':>7}")
    per_pos = []
    for pos, g in scored.groupby("position"):
        c = _corr(g.boltz.to_numpy(), g.foldx.to_numpy())
        bias = g.boltz.mean() - g.foldx.mean()
        per_pos.append(dict(position=pos, wt_aa=g.wt_aa.iloc[0], flagged=pos in FLAGGED,
                            mean_boltz=g.boltz.mean(), mean_foldx=g.foldx.mean(),
                            bias=bias, **c))
    per_pos = pd.DataFrame(per_pos).sort_values("position")
    for r in per_pos.itertuples(index=False):
        print(f"  {r.position:>4}{'*' if r.flagged else ' '}{r.wt_aa:>2} {r.n:>3} "
              f"{r.spearman:>+7.3f} {r.mean_boltz:>+7.2f} {r.mean_foldx:>+7.2f} "
              f"{r.bias:>+7.2f}")

    # --- does Boltz overestimate at the flagged sites, relative to the rest? ---
    fl, rest = per_pos[per_pos.flagged], per_pos[~per_pos.flagged]
    print(f"\n=== flagged vs the rest (bias = mean Boltz - mean FoldX) ===")
    print(f"  flagged ({len(fl)} sites) : bias {fl.bias.mean():+.2f}   "
          f"median rho {fl.spearman.median():+.3f}")
    print(f"  others  ({len(rest)} sites) : bias {rest.bias.mean():+.2f}   "
          f"median rho {rest.spearman.median():+.3f}")

    per_pos.to_csv(out / f"compare_foldx_per_position_{args.regime}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"compare_foldx_summary_{args.regime}.csv", index=False)
    scored[["mutation", "position", "wt_aa", "mut_aa", "boltz", "foldx"]].to_csv(
        out / f"compare_foldx_merged_{args.regime}.csv", index=False)

    _plots(scored, per_pos, out, args.regime)
    print(f"\nwrote -> {out}")


def _plots(scored, per_pos, out, regime):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.2))

    gly = scored.wt_aa == "G"
    a.scatter(scored.foldx[~gly], scored.boltz[~gly], s=12, alpha=0.45,
              color="#4C72B0", edgecolors="none", label="non-glycine")
    a.scatter(scored.foldx[gly], scored.boltz[gly], s=12, alpha=0.55,
              color="#C44E52", edgecolors="none", label="WT glycine")
    rho = spearmanr(scored.foldx, scored.boltz).statistic
    a.set_xscale("symlog", linthresh=5)
    a.set_xlabel(f"FoldX ΔΔG (kcal/mol, symlog — clash tail runs to +70)")
    a.set_ylabel(f"Boltz predicted ΔΔG ({regime})")
    a.set_title(f"Per mutation (n={len(scored)}), Spearman ρ={rho:+.3f}")
    a.axhline(0, color="0.6", lw=0.8); a.axvline(0, color="0.6", lw=0.8)
    a.legend(fontsize=8); a.grid(alpha=0.3)

    order = per_pos.sort_values("position")
    x = np.arange(len(order))
    b.plot(x, order.mean_boltz, "o-", ms=4, lw=1.2, label="Boltz", color="#4C72B0")
    b.plot(x, order.mean_foldx.clip(upper=15), "s-", ms=4, lw=1.2,
           label="FoldX (clipped at 15)", color="#DD8452")
    for i, r in enumerate(order.itertuples(index=False)):
        if r.flagged:
            b.axvspan(i-0.5, i+0.5, color="#C44E52", alpha=0.12)
    b.set_xticks(x)
    b.set_xticklabels([f"{r.wt_aa}{r.position}" for r in order.itertuples(index=False)],
                      rotation=90, fontsize=6)
    b.set_ylabel("mean ΔΔG over the 19 substitutions")
    b.set_title("Per position (shaded = flagged as overestimated)")
    b.legend(fontsize=8); b.grid(alpha=0.3)

    fig.tight_layout()
    p = out / f"figures/01_boltz_vs_foldx_{regime}.png"
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
