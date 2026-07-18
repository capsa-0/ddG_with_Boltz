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

Linear recalibration
--------------------
Transferred predictions are affinely miscalibrated (compressed toward the mean on an
out-of-range target). The eval also reports an out-of-sample (5-fold CV) linear
recalibration ``real ≈ a + b·pred`` — a ``pred_corrected`` column, ``corrected_*``
pooled metrics, and a ``scatter_corrected.png``. It lowers RMSE/MAE but, being affine,
leaves Pearson/Spearman unchanged.

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

from ddg.evaluation.labels import add_label_columns, feature_columns
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


def _cv_linear_calibration(y, pred, k: int = 5, seed: int = 0):
    """
    Out-of-sample affine recalibration of ``pred`` toward ``y``.

    Transferred predictions are affinely miscalibrated (compressed toward the mean on
    an out-of-range target dataset). Fit ``y ≈ a + b·pred`` by k-fold CV — each point
    is corrected by a fit that never saw it — so the corrected metrics aren't fitted on
    the points they score. This is a scale/offset fix only: it lowers RMSE/MAE but
    leaves Pearson/Spearman unchanged (correlation is invariant to affine transforms).

    Returns (corrected_predictions, global_intercept_a, global_slope_b).
    """
    from sklearn.model_selection import KFold
    cor = np.full(len(y), np.nan, dtype=float)
    for tr, te in KFold(n_splits=k, shuffle=True, random_state=seed).split(pred):
        b, a = np.polyfit(pred[tr], y[tr], 1)
        cor[te] = a + b * pred[te]
    b, a = np.polyfit(pred, y, 1)   # global coeffs, for reporting
    return cor, float(a), float(b)


def run_transfer(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    model_name: str = "mlp",
    drop_s: bool = False,
) -> dict:
    """
    Fit ``model_name`` on all of ``train_df`` and predict every row of ``test_df``.

    Returns a dict with the pooled metrics, the per-protein distribution, a
    ``sign_flipped`` flag, and a predictions DataFrame (``y``/``pred``/``unit``).
    """
    # Feature columns common to both tables (both are raw-Δz here, so this is the
    # full zdiag_/zpool_ set; the intersection guards against schema drift).
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

    # Out-of-sample linear recalibration (scale/offset fix; see helper).
    pred_cor, cal_a, cal_b = _cv_linear_calibration(y_te, pred)
    pooled_corrected = compute_metrics(y_te, pred_cor)

    labelled = add_label_columns(test_df).reset_index(drop=True)
    pred_df = pd.DataFrame({"y": y_te, "pred": pred, "pred_corrected": pred_cor,
                            "unit": labelled["protein"]})

    per_unit_rows = []
    for unit, g in pred_df.groupby("unit"):
        per_unit_rows.append({"unit": unit, **compute_metrics(g["y"], g["pred"])})
    per_protein = pd.DataFrame(per_unit_rows)
    dist = summarize(per_unit_rows)

    return {
        "pooled": pooled,
        "pooled_corrected": pooled_corrected,
        "cal_intercept": cal_a,
        "cal_slope": cal_b,
        "dist": dist,
        "sign_flipped": sign_flipped,
        "per_protein": per_protein,
        "predictions": pred_df,
        "n_features": len(keep),
        "model": model_name,
    }


def _scatter(pred_df: pd.DataFrame, pooled: dict, label_train: str,
             label_test: str, path: Path, *, pred_col: str = "pred",
             color: str = "#4C72B0", ylabel: str | None = None,
             title_suffix: str = "") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.6, 5.4))
    ax.scatter(pred_df["y"], pred_df[pred_col], s=10, alpha=0.4,
               color=color, edgecolors="none")
    # Scale each axis to its own data range (predictions span a much narrower band
    # than measured ΔΔG, so a shared range leaves the y-axis mostly empty). Draw the
    # y=x reference only across the range visible on both axes so it doesn't stretch
    # the limits back out.
    def _lim(v, pad=0.05):
        lo, hi = float(v.min()), float(v.max())
        m = (hi - lo) * pad or 1.0
        return lo - m, hi + m
    xlo, xhi = _lim(pred_df["y"])
    ylo, yhi = _lim(pred_df[pred_col])
    d0, d1 = max(xlo, ylo), min(xhi, yhi)
    ax.plot([d0, d1], [d0, d1], "--", color="0.4", lw=1)
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(ylo, yhi)
    ax.set_xlabel(f"Measured ΔΔG ({label_test})")
    ax.set_ylabel(ylabel or f"Predicted ΔΔG (trained on {label_train})")
    ax.set_title(f"Transfer {label_train} → {label_test}{title_suffix}\n"
                 f"r={pooled['pearson']:.3f}  ρ={pooled['spearman']:.3f}  "
                 f"RMSE={pooled['rmse']:.2f}  n={pooled['n']}")
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
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_df, test_df = _load(args.train), _load(args.test)
    res = run_transfer(train_df, test_df, model_name=args.model, drop_s=args.drop_s)

    summary = {
        "train": args.label_train, "test": args.label_test,
        "train_path": str(args.train), "test_path": str(args.test),
        "model": res["model"], "n_features": res["n_features"],
        "n_train": int(len(train_df)), "n_test": int(len(test_df)),
        "n_test_proteins": int(test_df["wt_id"].nunique()),
        "sign_flipped": res["sign_flipped"],
        "cal_intercept": res["cal_intercept"],
        "cal_slope": res["cal_slope"],
        **{f"pooled_{k}": v for k, v in res["pooled"].items()},
        **{f"corrected_{k}": v for k, v in res["pooled_corrected"].items()},
        **res["dist"],
    }
    (out / "transfer_summary.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).to_csv(out / "transfer_summary.csv", index=False)
    res["per_protein"].sort_values("n", ascending=False).to_csv(
        out / "per_protein.csv", index=False)
    res["predictions"].to_parquet(out / "predictions.parquet", index=False)
    _scatter(res["predictions"], res["pooled"], args.label_train,
             args.label_test, out / "scatter.png")
    _scatter(res["predictions"], res["pooled_corrected"], args.label_train,
             args.label_test, out / "scatter_corrected.png",
             pred_col="pred_corrected", color="#55934f",
             ylabel="Predicted ΔΔG — linear-corrected",
             title_suffix=" (linear-corrected, 5-fold CV)")

    p, pc = res["pooled"], res["pooled_corrected"]
    print(f"\n=== transfer {args.label_train} -> {args.label_test} "
          f"({res['model']}, sign_flipped={res['sign_flipped']}) ===")
    print(f"pooled:     r={p['pearson']:.3f}  rho={p['spearman']:.3f}  "
          f"RMSE={p['rmse']:.3f}  MAE={p['mae']:.3f}  n={p['n']}")
    print(f"lin-corr:   r={pc['pearson']:.3f}  rho={pc['spearman']:.3f}  "
          f"RMSE={pc['rmse']:.3f}  MAE={pc['mae']:.3f}  "
          f"(real ≈ {res['cal_intercept']:.3f} + {res['cal_slope']:.3f}·pred)")
    d = res["dist"]
    print(f"per-protein: r={d['pearson_mean']:.3f}±{d['pearson_sd']:.3f}  "
          f"(units={d['n_units']}, scored={d['n_units_scored']})")
    print(f"wrote -> {out}")


if __name__ == "__main__":
    main()
