"""
Module: error_curves
Description: How prediction error depends on the true ΔΔG — a diagnostic that works on
any predictions table with ``y`` (measured) and ``pred`` columns (transfer eval, or a
benchmark holdout's out-of-fold predictions).

We bin mutations by measured ΔΔG and, per bin, report:
  * bias         = mean(pred − y)      — systematic over/under-prediction
  * resid_std    = SD(pred − y)        — spread of the error (the "variance")
  * mae, rmse    = |error| summaries   — error magnitude

Two stacked panels share the measured-ΔΔG axis: (top) bias ± SD band, (bottom)
RMSE/MAE, with per-bin counts as faint bars. The classic signature of a model that
interpolates but does not extrapolate is a bias line that slopes down through zero
(over-predicts low ΔΔG, under-predicts high ΔΔG) with error magnitude rising toward
both tails.

``plot_density_error`` goes one step further and relates the error to the *training*
observation density in ΔΔG space (Gaussian KDE of the training labels): it plots
density and error along ΔΔG, and error-vs-density directly, quantifying the link with
the Spearman ρ between training density and error (points and bins). This isolates the
cause — error is low where training data is dense and rises where it is sparse.

CLI
---
    python -m ddg.evaluation.error_curves \
        --predictions data/processed/<exp>/.../predictions.parquet \
        --out <dir_or_png> --title "01 random holdout (HGB)" [--train-range -1,4]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def compute_error_curve(y, pred, bin_width: float = 0.5,
                        min_count: int = 8) -> pd.DataFrame:
    """Per-bin error stats vs measured ΔΔG (fixed-width bins, sparse bins dropped)."""
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    ok = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[ok], pred[ok]
    resid = pred - y
    lo = np.floor(y.min() / bin_width) * bin_width
    hi = np.ceil(y.max() / bin_width) * bin_width
    edges = np.arange(lo, hi + bin_width, bin_width)
    idx = np.digitize(y, edges)
    rows = []
    for b in range(1, len(edges)):
        m = idx == b
        n = int(m.sum())
        if n < min_count:
            continue
        r = resid[m]
        rows.append({
            "center": float((edges[b - 1] + edges[b]) / 2),
            "n": n,
            "bias": float(r.mean()),
            "resid_std": float(r.std()),
            "mae": float(np.abs(r).mean()),
            "rmse": float(np.sqrt((r ** 2).mean())),
        })
    return pd.DataFrame(rows)


def plot_error_vs_true(y, pred, out_path, title: str = "",
                       train_range: tuple[float, float] | None = None,
                       bin_width: float = 0.5, min_count: int = 8) -> pd.DataFrame:
    """Write the two-panel error-vs-ΔΔG figure; return the per-bin table."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = compute_error_curve(y, pred, bin_width=bin_width, min_count=min_count)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 6.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    c = curve["center"].to_numpy()

    if train_range is not None:
        for ax in (ax1, ax2):
            ax.axvspan(train_range[0], train_range[1], color="#4C72B0",
                       alpha=0.07, zorder=0)

    # Top: bias ± SD (the systematic error and its spread).
    ax1.axhline(0, color="0.6", lw=1)
    ax1.fill_between(c, curve["bias"] - curve["resid_std"],
                     curve["bias"] + curve["resid_std"], color="#4C72B0",
                     alpha=0.20, label="±1 SD of error")
    ax1.plot(c, curve["bias"], "-o", color="#14314f", ms=4,
             label="bias = mean(pred − measured)")
    ax1.set_ylabel("Prediction bias (kcal/mol)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_title(title or "Prediction error vs measured ΔΔG")

    # Bottom: error magnitude, with per-bin counts behind it.
    axc = ax2.twinx()
    axc.bar(c, curve["n"], width=bin_width * 0.9, color="0.85", zorder=0)
    axc.set_ylabel("n per bin", color="0.6")
    axc.tick_params(axis="y", colors="0.6")
    ax2.set_zorder(axc.get_zorder() + 1)
    ax2.patch.set_visible(False)
    ax2.plot(c, curve["rmse"], "-o", color="#C44E52", ms=4, label="RMSE")
    ax2.plot(c, curve["mae"], "-s", color="#DD8452", ms=4, label="MAE")
    ax2.set_ylabel("Error (kcal/mol)")
    ax2.set_xlabel("Measured ΔΔG (kcal/mol)")
    ax2.legend(loc="upper center", fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path / "error_vs_ddg.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return curve


def density_error_stats(train_ddg, y, pred, bin_width: float = 0.5,
                        min_count: int = 8):
    """
    Relate test error to *training* observation density in ΔΔG space.

    Estimates a Gaussian-KDE density of the training ΔΔG labels, evaluates it at each
    test point (and each error bin), and quantifies how error tracks density. Returns
    ``(bin_table, stats)`` where stats has the point-level and bin-level Spearman ρ
    between training density and error, plus the log-log RMSE-vs-density slope.
    """
    from scipy.stats import gaussian_kde, spearmanr
    train_ddg = np.asarray(train_ddg, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    kde = gaussian_kde(train_ddg)
    dens_pt = kde(y)
    abserr = np.abs(pred - y)
    point_rho = float(spearmanr(dens_pt, abserr).correlation)

    curve = compute_error_curve(y, pred, bin_width=bin_width, min_count=min_count)
    curve["train_density"] = kde(curve["center"].to_numpy())
    bin_rho = float(spearmanr(curve["train_density"], curve["rmse"]).correlation)
    good = curve["train_density"] > 0
    slope = (float(np.polyfit(np.log10(curve["train_density"][good]),
                              np.log10(curve["rmse"][good]), 1)[0])
             if int(good.sum()) >= 2 else float("nan"))
    return curve, {"density_error_spearman_points": point_rho,
                   "density_error_spearman_bins": bin_rho,
                   "loglog_slope": slope}


def plot_density_error(train_ddg, y, pred, out_path, title: str = "",
                       train_label: str = "training", bin_width: float = 0.5,
                       min_count: int = 8):
    """Two-panel figure: density & error along ΔΔG, and error-vs-density with ρ."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    train_ddg = np.asarray(train_ddg, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    curve, stats = density_error_stats(train_ddg, y, pred, bin_width, min_count)
    kde = gaussian_kde(train_ddg)
    xs = np.linspace(y.min(), y.max(), 400)
    c = curve["center"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.8))
    # Panel 1: density (twin) and error, along ΔΔG.
    axd = ax1.twinx()
    axd.fill_between(xs, kde(xs), color="#4C72B0", alpha=0.18, zorder=0)
    axd.plot(xs, kde(xs), color="#4C72B0", lw=1.5)
    axd.set_ylabel(f"{train_label} density", color="#4C72B0")
    axd.tick_params(axis="y", colors="#4C72B0")
    ax1.plot(c, curve["rmse"], "-o", color="#C44E52", ms=4, label="test RMSE")
    ax1.plot(c, curve["mae"], "-s", color="#DD8452", ms=3, label="test MAE")
    ax1.set_zorder(axd.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.set_xlabel("Measured ΔΔG (kcal/mol)")
    ax1.set_ylabel("Test error (kcal/mol)")
    ax1.set_title("Training density vs test error, along ΔΔG")
    ax1.legend(loc="upper center", fontsize=8)
    # Panel 2: error vs density (explicit), colored by ΔΔG.
    sc = ax2.scatter(curve["train_density"], curve["rmse"], c=c, cmap="viridis",
                     s=45, zorder=3, edgecolors="k", linewidths=0.3)
    ax2.set_xscale("log")
    good = curve["train_density"] > 0
    if int(good.sum()) >= 2:
        b, a = np.polyfit(np.log10(curve["train_density"][good]),
                          np.log10(curve["rmse"][good]), 1)
        xx = np.logspace(np.log10(curve["train_density"][good].min()),
                         np.log10(curve["train_density"].max()), 50)
        ax2.plot(xx, 10 ** (a + b * np.log10(xx)), "--", color="0.4", lw=1)
    ax2.set_xlabel(f"{train_label} density at that ΔΔG (log)")
    ax2.set_ylabel("Test RMSE (kcal/mol)")
    ax2.set_title("Test error vs training density")
    ax2.annotate(f"Spearman ρ (bins) = {stats['density_error_spearman_bins']:.2f}\n"
                 f"Spearman ρ (points) = {stats['density_error_spearman_points']:.2f}",
                 xy=(0.04, 0.06), xycoords="axes fraction", fontsize=9,
                 bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    fig.colorbar(sc, ax=ax2).set_label("Measured ΔΔG")
    ax2.grid(True, alpha=0.3, which="both")
    if title:
        fig.suptitle(title, y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    if out_path.suffix.lower() != ".png":
        out_path = out_path / "density_vs_error.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return curve, stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Prediction error vs measured ΔΔG")
    ap.add_argument("--predictions", required=True,
                    help="parquet/csv with columns y, pred")
    ap.add_argument("--out", required=True, help="output PNG (or dir)")
    ap.add_argument("--title", default="")
    ap.add_argument("--train-range", help="shade an in-range band 'lo,hi'")
    ap.add_argument("--train-ddg", help="parquet/csv with a 'ddg' column of the "
                    "training labels; enables the density-vs-error figure")
    ap.add_argument("--bin-width", type=float, default=0.5)
    ap.add_argument("--min-count", type=int, default=8)
    args = ap.parse_args()

    p = args.predictions
    df = pd.read_csv(p) if p.endswith(".csv") else pd.read_parquet(p)
    tr = None
    if args.train_range:
        lo, hi = (float(x) for x in args.train_range.split(","))
        tr = (lo, hi)
    out = Path(args.out)
    curve = plot_error_vs_true(df["y"], df["pred"], out, title=args.title,
                               train_range=tr, bin_width=args.bin_width,
                               min_count=args.min_count)
    print(curve.to_string(index=False))
    if args.train_ddg:
        tp = args.train_ddg
        tdf = pd.read_csv(tp) if tp.endswith(".csv") else pd.read_parquet(tp)
        dpath = (out.with_name("density_vs_error.png")
                 if out.suffix.lower() == ".png" else out / "density_vs_error.png")
        _, stats = plot_density_error(tdf["ddg"], df["y"], df["pred"], dpath,
                                      title=args.title, bin_width=args.bin_width,
                                      min_count=args.min_count)
        print("density-vs-error:", {k: round(v, 3) for k, v in stats.items()})
    print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
