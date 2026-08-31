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


def _wilson(k, n, z=1.96):
    """Wilson score interval — the binomial CI that stays sane at k/n near 0 or 1."""
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def percentile_shift(scored):
    """How each group sits relative to the percentile diagonal pct(Boltz) = pct(FoldX).

    Each method is ranked within its OWN spread, so the diagonal needs no fitting and
    is immune to the ~12x dynamic-range difference between the two. A point below it
    is one FoldX ranks higher, within its own distribution, than Boltz does.

    A fitted line would NOT do: the least-squares slope of Boltz on FoldX is ~0.055,
    so "below the line" degenerates into "low Boltz" and stops measuring disagreement
    at all (glycines then come out at 58% vs 56% for the rest, versus 78% vs 38% here).
    """
    gly = scored.wt_aa == "G"
    flag = scored.position.isin(FLAGGED)
    groups = {"all": pd.Series(True, index=scored.index),
              "glycine": gly,
              "non-glycine": ~gly,
              "flagged": flag,
              "flagged, non-glycine": flag & ~gly,
              "flagged, glycine": flag & gly,
              "rest (non-Gly, non-flagged)": ~gly & ~flag}
    rows = []
    for label, sel in groups.items():
        d = scored.delta[sel]
        k, n = int((d < 0).sum()), len(d)
        lo, hi = _wilson(k, n)
        rows.append(dict(group=label, n=n, n_below=k, pct_below=100 * k / n if n else np.nan,
                         ci_lo=100 * lo, ci_hi=100 * hi, median_delta=d.median()))
    return pd.DataFrame(rows)


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
    scored = df.dropna(subset=["foldx", "boltz"]).copy()
    # Rank each method within its own spread; their difference is the scale-free
    # measure of who is ranked higher by whom. Sums to ~0 by construction, so it is
    # a strictly relative statement between groups, never an accuracy claim.
    scored["pct_boltz"] = scored.boltz.rank(pct=True) * 100
    scored["pct_foldx"] = scored.foldx.rank(pct=True) * 100
    scored["delta"] = scored.pct_boltz - scored.pct_foldx
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

    shift = percentile_shift(scored)
    print("\n=== position relative to the percentile diagonal pct(Boltz) = pct(FoldX) ===")
    print("  below the diagonal = FoldX ranks it higher, within its own spread, than Boltz")
    print(f"  {'group':<30}{'below':>12}{'%':>8}{'95% CI':>16}{'median delta':>14}")
    for r in shift.itertuples(index=False):
        print(f"  {r.group:<30}{f'{r.n_below}/{r.n}':>12}{r.pct_below:>8.1f}"
              f"{f'[{r.ci_lo:.1f}, {r.ci_hi:.1f}]':>16}{r.median_delta:>+14.1f}")
    shift.to_csv(out / f"percentile_shift_{args.regime}.csv", index=False)

    per_pos.to_csv(out / f"compare_foldx_per_position_{args.regime}.csv", index=False)
    pd.DataFrame(rows).to_csv(out / f"compare_foldx_summary_{args.regime}.csv", index=False)
    scored[["mutation", "position", "wt_aa", "mut_aa", "boltz", "foldx",
            "pct_boltz", "pct_foldx", "delta"]].to_csv(
        out / f"compare_foldx_merged_{args.regime}.csv", index=False)

    _plots(scored, per_pos, shift, out, args.regime)
    print(f"\nwrote -> {out}")


def _plots(scored, per_pos, shift, out, regime):
    """Raw scatter, percentile-diagonal scatter, and the per-position comparison.

    The two annotations are orthogonal and overlap (3 of the 10 flagged positions are
    glycines), so they use different channels: glycine = colour/outline, flagged =
    shaded band + bold tick label.

    Layout: the two scatters share the top row, the per-position trace spans the
    bottom — it carries one tick per scanned position and needs the full width.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(14, 9.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=.3, wspace=.18)
    a = fig.add_subplot(gs[0, 0])
    c = fig.add_subplot(gs[0, 1])
    b = fig.add_subplot(gs[1, :])

    gly = (scored.wt_aa == "G").to_numpy()
    flag = scored.position.isin(FLAGGED).to_numpy()
    a.scatter(scored.foldx[~gly], scored.boltz[~gly], s=14, alpha=.45,
              color="#4C72B0", edgecolors="none", label=f"non-glycine (n={(~gly).sum()})")
    a.scatter(scored.foldx[gly], scored.boltz[gly], s=14, alpha=.5,
              color="#C44E52", edgecolors="none", label=f"WT glycine (n={gly.sum()})")
    # flagged mutations outlined on top, so they are visible in either colour
    a.scatter(scored.foldx[flag], scored.boltz[flag], s=42, facecolors="none",
              edgecolors="#B8860B", linewidths=.9,
              label=f"flagged position (n={flag.sum()})")
    rho = spearmanr(scored.foldx, scored.boltz).statistic
    a.set_xscale("symlog", linthresh=5)
    a.set_xlabel("FoldX ΔΔG (kcal/mol, symlog — clash tail runs to +69)")
    a.set_ylabel(f"Boltz predicted ΔΔG ({regime})")
    a.set_title(f"Per mutation (n={len(scored)}), Spearman ρ={rho:+.3f}")
    a.axhline(0, color="0.6", lw=.8); a.axvline(0, color="0.6", lw=.8)
    a.legend(fontsize=8, loc="upper left"); a.grid(alpha=.3)

    # --- percentile panel: same points, each method ranked within its own spread ---
    c.scatter(scored.pct_foldx[~gly], scored.pct_boltz[~gly], s=13, alpha=.40,
              color="#4C72B0", edgecolors="none")
    c.scatter(scored.pct_foldx[gly], scored.pct_boltz[gly], s=13, alpha=.45,
              color="#C44E52", edgecolors="none")
    c.scatter(scored.pct_foldx[flag], scored.pct_boltz[flag], s=40, facecolors="none",
              edgecolors="#B8860B", linewidths=.9)
    c.plot([0, 100], [0, 100], ls="--", lw=1.3, color="#333333", zorder=4)
    # Label each triangle where it belongs: below the diagonal is the FoldX-higher side.
    c.text(97, 12, "below: FoldX ranks it higher", ha="right", va="center", fontsize=8.5,
           style="italic", color="#555555")
    c.text(6, 94, "above: Boltz ranks it higher", ha="left", va="center", fontsize=8.5,
           style="italic", color="#555555")
    pct = shift.set_index("group").pct_below
    box = "\n".join([f"{lab:<22}{pct[key]:5.1f}%" for lab, key in (
        ("WT glycine", "glycine"),
        ("  · also flagged", "flagged, glycine"),
        ("flagged, non-glycine", "flagged, non-glycine"),
        ("rest", "rest (non-Gly, non-flagged)"),
        ("all", "all"))])
    c.text(.03, .74, "% below the diagonal\n" + box, transform=c.transAxes,
           va="top", ha="left", fontsize=8, family="monospace",
           bbox=dict(boxstyle="round,pad=.45", fc="white", ec="#BBBBBB", alpha=.92))
    c.set_xlabel("FoldX ΔΔG — percentile within FoldX")
    c.set_ylabel(f"Boltz ΔΔG ({regime}) — percentile within Boltz")
    c.set_title("Same points, each method ranked within its own spread", fontsize=10)
    c.set_xlim(-2, 102); c.set_ylim(-2, 102); c.grid(alpha=.3)

    o = per_pos.sort_values("position").reset_index(drop=True)
    x = np.arange(len(o))
    is_g = (o.wt_aa == "G").to_numpy()
    for i, r in enumerate(o.itertuples(index=False)):
        if r.flagged:
            b.axvspan(i - .5, i + .5, color="#DDAA33", alpha=.22, zorder=0)
    b.plot(x, o.mean_boltz, "o-", ms=4, lw=1.2, label="Boltz", color="#4C72B0", zorder=3)
    b.plot(x, o.mean_foldx.clip(upper=15), "s-", ms=4, lw=1.2, color="#DD8452",
           label="FoldX (clipped at 15)", zorder=3)
    # glycine positions get a filled marker along the baseline
    b.scatter(x[is_g], np.full(is_g.sum(), -1.2), marker="^", s=22, color="#333333",
              zorder=4, clip_on=False, label=f"WT glycine ({is_g.sum()} of {len(o)})")
    b.set_xticks(x)
    b.set_xticklabels([f"{r.wt_aa}{r.position}" + ("*" if r.flagged else "")
                       for r in o.itertuples(index=False)], rotation=90, fontsize=6)
    for tick, fl in zip(b.get_xticklabels(), o.flagged):
        if fl:
            tick.set_color("#B8860B"); tick.set_fontweight("bold")
    b.set_ylabel("mean ΔΔG over the substitutions at that position")
    b.set_title("Per position  (* / shaded = flagged · ▲ = wild-type glycine)", fontsize=10)
    b.legend(fontsize=8); b.grid(alpha=.3)

    p = out / f"figures/01_boltz_vs_foldx_{regime}.png"
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=150)
    plt.close(fig)

    plot_percentile_shift(shift, out, regime)


def plot_percentile_shift(shift, out, regime):
    """Forest plot of the per-group percentile shift.

    The group table is where the actual claim lives -- whether the flagged positions
    are shifted once glycines are taken out of them -- and it was previously only
    printed, never drawn, so the headline had no picture to stand on. One measure
    (% of mutations FoldX ranks higher) on one axis, 50 % marking no systematic
    disagreement; a group whose interval clears 50 % is a real shift.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = ["glycine", "flagged", "all", "flagged, non-glycine",
             "non-glycine", "rest (non-Gly, non-flagged)", "flagged, glycine"]
    d = shift.set_index("group").loc[[g for g in order if g in set(shift.group)]].reset_index()
    # colour encodes the reading, not the label: does the interval clear 50 %?
    ORANGE, BLUE, GREY = "#D95F02", "#1F6FB4", "#8b949e"
    cols = [ORANGE if lo > 50 else BLUE if hi < 50 else GREY
            for lo, hi in zip(d.ci_lo, d.ci_hi)]
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    y = np.arange(len(d))[::-1]
    ax.axvline(50, color="#1a1a1a", lw=1.1, ls="--", alpha=.6, zorder=1)
    for yy, r, c in zip(y, d.itertuples(index=False), cols):
        ax.plot([r.ci_lo, r.ci_hi], [yy, yy], color=c, lw=2.4,
                solid_capstyle="round", zorder=2)
        ax.plot(r.pct_below, yy, "o", ms=8, color=c, mec="white", mew=1.4, zorder=3)
        ax.text(r.ci_hi + 1.2, yy, f"{r.pct_below:.1f}%  (n={r.n})",
                va="center", fontsize=8.5, color="#1a1a1a")
    ax.set_yticks(y); ax.set_yticklabels(d.group, fontsize=9)
    ax.set_xlabel("% of mutations FoldX ranks as more destabilizing than Boltz does")
    ax.set_xlim(30, 108); ax.set_ylim(-0.7, len(d) - 0.3)
    ax.xaxis.grid(True, color="#c9d1d9", lw=.6); ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_title("Where the two methods disagree, by group\n"
                 "dashed line = 50 %, i.e. no systematic disagreement; "
                 "bars are 95 % CI", fontsize=10, loc="left")
    ax.text(0.0, -0.19, "orange = FoldX systematically harsher   ·   "
                        "blue = Boltz systematically harsher   ·   "
                        "grey = interval covers 50 %, no separation",
            transform=ax.transAxes, fontsize=8, color="#555555")
    p2 = out / f"figures/05_percentile_shift_{regime}.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
