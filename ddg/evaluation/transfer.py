"""
Module: transfer
Description: Cross-dataset transfer evaluation for the raw-Δz ΔΔG predictor.

Unlike ddg.evaluation.benchmark (which does *within-dataset* group holdouts), this
trains one model on the *entire* source features table and predicts every row of an
*independent* target table, then reports how well the source-trained model transfers.
It is the tool behind results/05_cross_dataset_fireprot (train on Tsuboyama, test on
FireProt).

Both tables must be raw-Δz feature tables (``zdiag_*`` + ``zpool_*`` columns, plus
``wt_id``/``mutation``/``ddg``). The model is the benchmark's pipeline
(median-impute -> standardize -> estimator); ``hgb`` by default.

In-range vs out-of-range
------------------------
The model is trained on Tsuboyama, whose ΔΔG labels occupy a narrow range (~−1 to 4
kcal/mol, the central ~98% of the training distribution). FireProt has mutations well
outside that, which are genuine extrapolation. The eval scores the transfer separately
for **in-range** and **out-of-range** test mutations (by measured ΔΔG vs the training
range), and the scatter shades the in-range band and colors out-of-range points.

Sign convention
---------------
ΔΔG sign conventions can differ between datasets (FireProt ``ddG`` vs Tsuboyama
``ddg``). Pearson/Spearman are sign-agnostic in magnitude, but a negative pooled
Pearson means the two datasets define "destabilizing" with opposite signs, so the
error metrics (RMSE/MAE) are meaningless until predictions are flipped. When the
pooled Pearson comes out negative this tool flips the sign of the predictions,
recomputes, and records ``sign_flipped: true`` in the summary.

CLI
---
    python -m ddg.evaluation.transfer \
        --train data/processed/tsuboyama_bench_fast/rawz_features.parquet \
        --test  data/processed/fireprot_le200/features_summary.parquet \
        --out   data/processed/fireprot_le200/transfer_from_tsuboyama \
        --model hgb --label-train Tsuboyama --label-test FireProt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ddg.evaluation.labels import (TRANSFER_BLOCKS, add_label_columns,
                                   block_columns, feature_columns)
from ddg.evaluation.metrics import compute_metrics, summarize
from ddg.evaluation.models import make_model

logger = logging.getLogger(__name__)


def _load(path: str | Path) -> pd.DataFrame:
    path = str(path)
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


def _matrix(df: pd.DataFrame, feat_cols: list[str]) -> np.ndarray:
    """Feature matrix on a fixed column set; inf -> NaN (imputer handles NaN)."""
    return df.reindex(columns=feat_cols).replace([np.inf, -np.inf], np.nan) \
             .to_numpy(dtype=float)


def _train_ddg_range(train_df: pd.DataFrame, pct: float = 1.0):
    """In-range boundary = [pct, 100-pct] percentile of the training ΔΔG labels.

    Percentiles (not min/max) so a few extreme training points don't widen the range;
    the default 1st/99th ≈ the '−1 to 4 kcal/mol' effective Tsuboyama range.
    """
    y = train_df["ddg"].to_numpy(dtype=float)
    return float(np.percentile(y, pct)), float(np.percentile(y, 100 - pct))


def run_transfer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "mlp",
    drop_s: bool = False,
    train_range: tuple[float, float] | None = None,
) -> dict:
    """
    Fit ``model_name`` on all of ``train_df`` and predict every row of ``test_df``.

    ``train_range`` = (lo, hi) defining the in-range band on measured ΔΔG; test points
    outside it are out-of-range (extrapolation). Defaults to the training ΔΔG's
    1st/99th percentile.

    Returns a dict with the pooled metrics, the per-protein distribution, a
    ``sign_flipped`` flag, the in/out-of-range split, and a predictions DataFrame.
    """
    # Readout. CHANGED 2026-08-27 (results/14): this module trains on one corpus and
    # tests on another, so it takes the TRANSFER readout — the pair-track diagonal
    # alone — rather than every numeric column. On two blind corpora the diagonal
    # (128d) matches every 256-d construction, while the pooled levels wtz/mtz carry a
    # corpus-specific per-protein offset costing -0.141 [-0.241,-0.020] r. Tables built
    # before the all-blocks default may lack zdiag_*; fall back to the old behaviour
    # (every shared numeric column) with a warning rather than failing.
    try:
        shared = set(block_columns(train_df, TRANSFER_BLOCKS)) & \
                 set(block_columns(test_df, TRANSFER_BLOCKS))
        feat_cols = sorted(shared, key=lambda c: int(c.rsplit("_", 1)[1]))
        ftr = fte = shared
    except ValueError as exc:
        logger.warning("transfer readout unavailable (%s); falling back to all shared "
                       "numeric columns — see results/14 on why this transfers worse", exc)
        ftr = set(feature_columns(train_df, drop_s=drop_s))
        fte = set(feature_columns(test_df, drop_s=drop_s))
        feat_cols = sorted(ftr & fte)
    if not feat_cols:
        raise ValueError("no shared feature columns between train and test tables")
    if ftr != fte:
        logger.warning("feature sets differ: train-only=%d test-only=%d shared=%d",
                       len(ftr - fte), len(fte - ftr), len(feat_cols))

    X_tr = _matrix(train_df, feat_cols)
    y_tr = train_df["ddg"].to_numpy(dtype=float)
    # Drop columns that are entirely non-finite in train (the imputer can't learn a
    # median for them); keep the same columns on the test side for alignment.
    keep = [c for c, ok in zip(feat_cols, np.isfinite(X_tr).any(axis=0)) if ok]
    if len(keep) != len(feat_cols):
        logger.warning("dropped %d all-nonfinite train columns", len(feat_cols) - len(keep))
    X_tr = _matrix(train_df, keep)
    X_te = _matrix(test_df, keep)
    y_te = test_df["ddg"].to_numpy(dtype=float)

    logger.info("train n=%d, test n=%d, features=%d, model=%s",
                len(train_df), len(test_df), len(keep), model_name)

    model = make_model(model_name)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_te)

    pooled = compute_metrics(y_te, pred)
    sign_flipped = False
    if np.isfinite(pooled["pearson"]) and pooled["pearson"] < 0:
        logger.warning("pooled Pearson r=%.3f < 0 -> flipping prediction sign "
                       "(opposite ΔΔG conventions between datasets)", pooled["pearson"])
        pred = -pred
        pooled = compute_metrics(y_te, pred)
        sign_flipped = True

    # In-range vs out-of-range split, by measured ΔΔG against the training range.
    lo, hi = train_range if train_range is not None else _train_ddg_range(train_df)
    in_range = (y_te >= lo) & (y_te <= hi)
    by_range = {
        "lo": float(lo), "hi": float(hi),
        "in": compute_metrics(y_te[in_range], pred[in_range]),
        "out": compute_metrics(y_te[~in_range], pred[~in_range]),
        "out_below": compute_metrics(y_te[y_te < lo], pred[y_te < lo]),
        "out_above": compute_metrics(y_te[y_te > hi], pred[y_te > hi]),
    }
    logger.info("in-range [%.2f,%.2f]: n_in=%d n_out=%d (below=%d above=%d)",
                lo, hi, int(in_range.sum()), int((~in_range).sum()),
                int((y_te < lo).sum()), int((y_te > hi).sum()))

    labelled = add_label_columns(test_df).reset_index(drop=True)
    pred_df = pd.DataFrame({"y": y_te, "pred": pred, "in_range": in_range,
                            "unit": labelled["protein"]})

    per_unit_rows = []
    for unit, g in pred_df.groupby("unit"):
        per_unit_rows.append({"unit": unit, **compute_metrics(g["y"], g["pred"])})
    per_protein = pd.DataFrame(per_unit_rows)
    dist = summarize(per_unit_rows)

    return {
        "pooled": pooled,
        "by_range": by_range,
        "dist": dist,
        "sign_flipped": sign_flipped,
        "per_protein": per_protein,
        "predictions": pred_df,
        "n_features": len(keep),
        "model": model_name,
    }


def _scatter(pred_df: pd.DataFrame, by_range: dict, label_train: str,
             label_test: str, path: Path) -> None:
    """Predicted vs measured ΔΔG, in-range band shaded and out-of-range points colored."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = by_range["lo"], by_range["hi"]
    ins = pred_df["in_range"].to_numpy()
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    # Scale each axis to its own data range (predictions span a much narrower band
    # than measured ΔΔG, so a shared range leaves the y-axis mostly empty).
    def _lim(v, pad=0.05):
        a, b = float(v.min()), float(v.max())
        m = (b - a) * pad or 1.0
        return a - m, b + m
    xlo, xhi = _lim(pred_df["y"])
    ylo, yhi = _lim(pred_df["pred"])
    ax.axvspan(lo, hi, color="#4C72B0", alpha=0.07, zorder=0)
    for x in (lo, hi):
        ax.axvline(x, color="#4C72B0", ls=":", lw=1.2, zorder=1)
    # in-range shows r (meaningful); out-of-range shows RMSE only — its pooled r is a
    # two-cluster artifact (the two tails sit far apart), so per-tail r goes in the table.
    m_in, m_out = by_range["in"], by_range["out"]
    ax.scatter(pred_df["y"][ins], pred_df["pred"][ins], s=10, alpha=0.4,
               color="#4C72B0", edgecolors="none",
               label=f"in-range (n={m_in['n']}):  r={m_in['pearson']:.2f}, RMSE={m_in['rmse']:.2f}")
    ax.scatter(pred_df["y"][~ins], pred_df["pred"][~ins], s=12, alpha=0.55,
               color="#C44E52", edgecolors="none",
               label=f"out-of-range (n={m_out['n']}):  RMSE={m_out['rmse']:.2f}, MAE={m_out['mae']:.2f}")
    d0, d1 = max(xlo, ylo), min(xhi, yhi)
    ax.plot([d0, d1], [d0, d1], "--", color="0.4", lw=1, zorder=1)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"Measured ΔΔG ({label_test})")
    ax.set_ylabel(f"Predicted ΔΔG (trained on {label_train})")
    ax.set_title(f"Transfer {label_train} → {label_test}\n"
                 f"training range [{lo:.1f}, {hi:.1f}] kcal/mol shaded")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    ap = argparse.ArgumentParser(description="Cross-dataset ΔΔG transfer eval")
    ap.add_argument("--train", required=True, help="source features table (parquet/csv)")
    ap.add_argument("--test", required=True, help="target features table (parquet/csv)")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model", default="mlp", choices=["hgb", "svr", "ridge", "mlp"])
    ap.add_argument("--drop-s", action="store_true", help="exclude s-derived features")
    ap.add_argument("--label-train", default="train")
    ap.add_argument("--label-test", default="test")
    ap.add_argument("--train-range", help="in-range band 'lo,hi' on measured ΔΔG "
                    "(default: 1st/99th percentile of training ΔΔG)")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_df, test_df = _load(args.train), _load(args.test)
    tr_range = None
    if args.train_range:
        lo, hi = (float(x) for x in args.train_range.split(","))
        tr_range = (lo, hi)
    res = run_transfer(train_df, test_df, model_name=args.model, drop_s=args.drop_s,
                       train_range=tr_range)

    br = res["by_range"]
    summary = {
        "train": args.label_train, "test": args.label_test,
        "train_path": str(args.train), "test_path": str(args.test),
        "model": res["model"], "n_features": res["n_features"],
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_test_proteins": int(test_df["wt_id"].nunique()),
        "sign_flipped": res["sign_flipped"],
        "range_lo": br["lo"], "range_hi": br["hi"],
        **{f"pooled_{k}": v for k, v in res["pooled"].items()},
        **{f"in_{k}": v for k, v in br["in"].items()},
        **{f"out_{k}": v for k, v in br["out"].items()},
        **{f"out_below_{k}": v for k, v in br["out_below"].items()},
        **{f"out_above_{k}": v for k, v in br["out_above"].items()},
        **res["dist"],
    }
    (out / "transfer_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(out / "transfer_summary.csv", index=False)
    res["per_protein"].sort_values("n", ascending=False).to_csv(
        out / "per_protein.csv", index=False)
    res["predictions"].to_parquet(out / "predictions.parquet", index=False)
    _scatter(res["predictions"], br, args.label_train, args.label_test,
             out / "scatter.png")
    from ddg.evaluation.error_curves import plot_error_vs_true, plot_density_error
    plot_error_vs_true(
        res["predictions"]["y"], res["predictions"]["pred"], out / "error_vs_ddg.png",
        title=f"{args.label_train} → {args.label_test}: error vs measured ΔΔG",
        train_range=(br["lo"], br["hi"]))
    _, dstats = plot_density_error(
        train_df["ddg"], res["predictions"]["y"], res["predictions"]["pred"],
        out / "density_vs_error.png", train_label=args.label_train,
        title=f"{args.label_train} → {args.label_test}: training density vs test error")
    summary.update(dstats)
    (out / "transfer_summary.json").write_text(json.dumps(summary, indent=2))

    p = res["pooled"]
    print(f"\n=== transfer {args.label_train} -> {args.label_test} "
          f"({res['model']}, sign_flipped={res['sign_flipped']}) ===")
    print(f"pooled:        r={p['pearson']:.3f}  rho={p['spearman']:.3f}  "
          f"RMSE={p['rmse']:.3f}  MAE={p['mae']:.3f}  n={p['n']}")
    print(f"in-range [{br['lo']:.2f},{br['hi']:.2f}]: r={br['in']['pearson']:.3f}  "
          f"rho={br['in']['spearman']:.3f}  RMSE={br['in']['rmse']:.3f}  "
          f"MAE={br['in']['mae']:.3f}  n={br['in']['n']}")
    print(f"out-of-range:  r={br['out']['pearson']:.3f}  rho={br['out']['spearman']:.3f}  "
          f"RMSE={br['out']['rmse']:.3f}  MAE={br['out']['mae']:.3f}  n={br['out']['n']} "
          f"(below n={br['out_below']['n']} RMSE={br['out_below']['rmse']:.2f} | "
          f"above n={br['out_above']['n']} RMSE={br['out_above']['rmse']:.2f})")
    d = res["dist"]
    print(f"per-protein: r={d['pearson_mean']:.3f}±{d['pearson_sd']:.3f}  "
          f"(units={d['n_units']}, scored={d['n_units_scored']})")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
