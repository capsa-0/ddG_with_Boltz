"""
Module: metrics
Description: Regression metrics for ΔΔG predictions, robust to degenerate folds.
"""

import numpy as np
from scipy.stats import pearsonr, spearmanr


def compute_metrics(y_true, y_pred) -> dict:
    """
    Pearson r, Spearman rho, RMSE, MAE and N for one (true, pred) pair.

    Correlations are NaN when N < 2 or either array is constant (undefined), so
    callers can drop them from means without crashing.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    ok = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[ok], y_pred[ok]
    n = int(y_true.size)

    out = {"n": n, "pearson": np.nan, "spearman": np.nan, "rmse": np.nan, "mae": np.nan}
    if n == 0:
        return out
    out["rmse"] = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    out["mae"] = float(np.mean(np.abs(y_true - y_pred)))
    if n >= 2 and y_true.std() > 0 and y_pred.std() > 0:
        out["pearson"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman"] = float(spearmanr(y_true, y_pred)[0])
    return out


def summarize(per_unit_metrics: list[dict]) -> dict:
    """
    Aggregate a list of per-unit metric dicts into mean±SD (ignoring NaN units).

    Used for the per-protein / per-cluster distributions where each unit yields
    its own metrics.
    """
    summary = {}
    for key in ("pearson", "spearman", "rmse", "mae"):
        vals = np.array([m[key] for m in per_unit_metrics], dtype=float)
        vals = vals[np.isfinite(vals)]
        summary[f"{key}_mean"] = float(np.mean(vals)) if vals.size else np.nan
        summary[f"{key}_sd"] = float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan
    summary["n_units"] = len(per_unit_metrics)
    summary["n_units_scored"] = int(np.isfinite(
        [m["pearson"] for m in per_unit_metrics]).sum())
    return summary
