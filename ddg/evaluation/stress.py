"""
Module: stress
Description: Stress tests for the ΔΔG predictor, beyond the single-axis holdouts
of ddg.evaluation. Both run on the raw-Δz feature table (features_summary.parquet
or rawz_features.parquet) with the benchmark's HGB pipeline.

Tests
-----
1. extrapolation — Train ONLY on mild mutations (|ΔΔG| < `mild`) and test on the
   strongly destabilizing tail (ΔΔG > `tail`). Contrasts the tail against an
   in-distribution baseline (held-out mild mutations) to expose regression to the
   mean: the predicted-vs-actual fit slope collapses on the tail. Reports r / RMSE
   / MAE / slope / range-coverage and a predicted-vs-actual figure.

2. learning_curve — Grouped by protein (test proteins never seen in training),
   train on 10/25/50/100 % of the available training proteins and plot pooled r
   (and RMSE) vs. the number of training proteins. Shows data efficiency /
   saturation. Averaged over several random protein subsamples per fraction.

CLI
---
    python -m ddg.evaluation.stress extrapolation --parquet <p> --out <dir>
    python -m ddg.evaluation.stress learning_curve --parquet <p> --out <dir>
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ddg.evaluation.models import make_model, build_xy
from ddg.evaluation.metrics import compute_metrics

logger = logging.getLogger(__name__)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def _fit_slope(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Slope b of the least-squares fit pred = a + b·true.

    b < 1 means predictions are compressed toward the training mean (regression
    to the mean); b ≈ 1 means the model tracks the true magnitude.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    if y_true.size < 2 or y_true.std() == 0:
        return float("nan")
    return float(np.polyfit(y_true, y_pred, 1)[0])


# --------------------------------------------------------------------------- #
# Test 1: extrapolation to the destabilizing tail
# --------------------------------------------------------------------------- #
def run_extrapolation(df: pd.DataFrame, out_dir: Path, model: str = "hgb",
                      mild: float = 1.0, tail: float = 2.0,
                      in_dist_frac: float = 0.2, seed: int = 0) -> dict:
    """Train on mild mutations, test on the destabilizing tail + in-dist baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    is_mild = df["ddg"].abs() < mild
    is_tail = df["ddg"] > tail
    mild_df = df[is_mild]
    tail_df = df[is_tail]

    # Hold out a random slice of the mild set as the in-distribution baseline.
    n_hold = int(round(in_dist_frac * len(mild_df)))
    perm = rng.permutation(len(mild_df))
    hold_idx = mild_df.index[perm[:n_hold]]
    train_df = mild_df.drop(index=hold_idx)
    indist_df = mild_df.loc[hold_idx]

    feat_cols = [c for c in df.columns if c.startswith(("zdiag_", "zpool_", "sdim_"))]
    Xtr, ytr, kept = build_xy(train_df, feat_cols=feat_cols)
    pipe = make_model(model)
    pipe.fit(Xtr, ytr)
    logger.info("extrapolation: trained on %d mild mutations (%d features)",
                len(train_df), len(kept))

    def _eval(sub: pd.DataFrame) -> dict:
        X = sub[kept].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        y = sub["ddg"].to_numpy(float)
        pred = pipe.predict(X)
        m = compute_metrics(y, pred)
        m["slope"] = _fit_slope(y, pred)
        m["true_max"] = float(np.max(y)) if len(y) else float("nan")
        m["pred_max"] = float(np.max(pred)) if len(pred) else float("nan")
        # Range coverage: how much of the true span the predictions reach.
        m["range_coverage"] = (m["pred_max"] - float(np.min(pred))) / \
            (m["true_max"] - float(np.min(y))) if len(y) and np.ptp(y) > 0 else float("nan")
        return m, pred

    indist_m, indist_pred = _eval(indist_df)
    tail_m, tail_pred = _eval(tail_df)

    result = {
        "model": model, "mild_threshold": mild, "tail_threshold": tail,
        "n_train_mild": len(train_df),
        "in_distribution": {"split": f"held-out mild (|ΔΔG|<{mild})", **indist_m},
        "tail": {"split": f"ΔΔG>{tail}", **tail_m},
    }
    with open(out_dir / "extrapolation_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    pd.DataFrame([
        {"split": "in_distribution", **indist_m},
        {"split": "tail", **tail_m},
    ]).to_csv(out_dir / "extrapolation_summary.csv", index=False)

    _plot_extrapolation(indist_df["ddg"].to_numpy(), indist_pred,
                        tail_df["ddg"].to_numpy(), tail_pred,
                        indist_m, tail_m, mild, tail,
                        out_dir / "extrapolation_pred_vs_actual.png")
    logger.info("extrapolation: in-dist r=%.3f slope=%.3f | tail r=%.3f slope=%.3f "
                "coverage=%.2f", indist_m["pearson"], indist_m["slope"],
                tail_m["pearson"], tail_m["slope"], tail_m["range_coverage"])
    return result


def _plot_extrapolation(y_in, p_in, y_tail, p_tail, m_in, m_tail,
                        mild, tail, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2), sharex=True, sharey=True)
    lo = min(y_in.min(), y_tail.min(), p_in.min(), p_tail.min()) - 0.3
    hi = max(y_in.max(), y_tail.max(), p_in.max(), p_tail.max()) + 0.3
    for ax, (y, p, m, title, color) in zip(axes, [
        (y_in, p_in, m_in, f"In-distribution (held-out |ΔΔG|<{mild})", "#4C72B0"),
        (y_tail, p_tail, m_tail, f"Extrapolation tail (ΔΔG>{tail})", "#C44E52"),
    ]):
        ax.scatter(y, p, s=8, alpha=0.25, color=color, edgecolors="none")
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        b, a = np.polyfit(y, p, 1)
        xs = np.array([y.min(), y.max()])
        ax.plot(xs, a + b * xs, color="k", lw=1.6, label=f"fit (slope={b:.2f})")
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Experimental ΔΔG (kcal/mol)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.text(0.04, 0.96, f"n={m['n']}\nr={m['pearson']:.3f}\nRMSE={m['rmse']:.2f}",
                transform=ax.transAxes, va="top", ha="left", fontsize=9,
                bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))
        ax.legend(loc="lower right", fontsize=8, framealpha=0.9)
    axes[0].set_ylabel("Predicted ΔΔG (kcal/mol)")
    fig.suptitle("Extrapolation to the destabilizing tail — trained on mild "
                 "mutations only", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Test 2: learning curve over number of training proteins
# --------------------------------------------------------------------------- #
def run_learning_curve(df: pd.DataFrame, out_dir: Path, model: str = "hgb",
                       fractions=(0.1, 0.25, 0.5, 1.0), n_folds: int = 5,
                       n_seeds: int = 3, seed: int = 0) -> pd.DataFrame:
    """Pooled r vs. number of training proteins (test proteins held out)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = [c for c in df.columns if c.startswith(("zdiag_", "zpool_", "sdim_"))]
    groups = df["wt_id"].to_numpy()
    X_all, y_all, kept = build_xy(df, feat_cols=feat_cols)
    gkf = GroupKFold(n_splits=n_folds)
    folds = list(gkf.split(X_all, y_all, groups))

    rows = []
    for frac in fractions:
        # One pooled prediction vector per seed, then average the pooled metric.
        seed_metrics = []
        n_train_proteins = []
        seeds = [seed] if frac >= 1.0 else range(seed, seed + n_seeds)
        for s in seeds:
            rng = np.random.default_rng(s)
            pooled_true, pooled_pred = [], []
            n_prot_this = []
            for tr_idx, te_idx in folds:
                train_prot = np.unique(groups[tr_idx])
                k = max(1, int(round(frac * len(train_prot))))
                chosen = set(rng.choice(train_prot, size=k, replace=False))
                n_prot_this.append(len(chosen))
                sel = np.array([g in chosen for g in groups[tr_idx]])
                pipe = make_model(model)
                pipe.fit(X_all[tr_idx][sel], y_all[tr_idx][sel])
                pooled_true.append(y_all[te_idx])
                pooled_pred.append(pipe.predict(X_all[te_idx]))
            m = compute_metrics(np.concatenate(pooled_true),
                                np.concatenate(pooled_pred))
            seed_metrics.append(m)
            n_train_proteins.append(np.mean(n_prot_this))
        rows.append({
            "fraction": frac,
            "n_train_proteins": float(np.mean(n_train_proteins)),
            "pooled_pearson_mean": float(np.mean([m["pearson"] for m in seed_metrics])),
            "pooled_pearson_sd": float(np.std([m["pearson"] for m in seed_metrics], ddof=1)
                                       if len(seed_metrics) > 1 else 0.0),
            "pooled_rmse_mean": float(np.mean([m["rmse"] for m in seed_metrics])),
            "pooled_mae_mean": float(np.mean([m["mae"] for m in seed_metrics])),
            "n_seeds": len(seed_metrics),
        })
        logger.info("learning_curve: frac=%.2f (~%d proteins) pooled r=%.3f",
                    frac, rows[-1]["n_train_proteins"], rows[-1]["pooled_pearson_mean"])

    curve = pd.DataFrame(rows)
    curve.to_csv(out_dir / "learning_curve.csv", index=False)
    _plot_learning_curve(curve, out_dir / "learning_curve.png")
    return curve


def _plot_learning_curve(curve: pd.DataFrame, path: Path) -> None:
    fig, ax1 = plt.subplots(figsize=(7.5, 5))
    x = curve["n_train_proteins"]
    ebar = ax1.errorbar(x, curve["pooled_pearson_mean"], yerr=curve["pooled_pearson_sd"],
                        marker="o", color="#4C72B0", lw=1.8, capsize=3,
                        label="pooled Pearson r")
    ax1.set_xlabel("Number of training proteins")
    ax1.set_ylabel("Pooled Pearson r", color="#4C72B0")
    ax1.tick_params(axis="y", labelcolor="#4C72B0")
    ax1.grid(True, alpha=0.3)
    for xi, yi in zip(x, curve["pooled_pearson_mean"]):
        ax1.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                     xytext=(0, 8), fontsize=8, ha="center")
    ax2 = ax1.twinx()
    rmse_line, = ax2.plot(x, curve["pooled_rmse_mean"], marker="s", color="#C44E52",
                          lw=1.4, ls="--", label="pooled RMSE")
    ax2.set_ylabel("Pooled RMSE (kcal/mol)", color="#C44E52")
    ax2.tick_params(axis="y", labelcolor="#C44E52")
    # Explicit handles so the errorbar's cap/bar lines don't leak into the legend.
    ax1.legend([ebar, rmse_line], ["pooled Pearson r", "pooled RMSE"],
               loc="center right", fontsize=9)
    ax1.set_title("Learning curve — pooled r vs. training-set size (proteins held out)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def _load(parquet: str) -> pd.DataFrame:
    p = Path(parquet)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    if "ddg" not in df.columns or "wt_id" not in df.columns:
        raise SystemExit(f"{parquet} must have 'ddg' and 'wt_id' columns")
    return df


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ap = argparse.ArgumentParser(description="ΔΔG predictor stress tests")
    ap.add_argument("test", choices=["extrapolation", "learning_curve"])
    ap.add_argument("--parquet", required=True, help="raw-Δz feature table")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model", default="hgb")
    ap.add_argument("--mild", type=float, default=1.0)
    ap.add_argument("--tail", type=float, default=2.0)
    args = ap.parse_args()

    df = _load(args.parquet)
    out = Path(args.out)
    if args.test == "extrapolation":
        run_extrapolation(df, out, model=args.model, mild=args.mild, tail=args.tail)
    else:
        run_learning_curve(df, out, model=args.model)


if __name__ == "__main__":
    main()
