"""
Module: benchmark
Description: Run the full holdout suite over a features table and write tables.

Public entry point ``run_benchmark`` returns a BenchmarkResults object and (by
default) writes:
  <out_dir>/benchmark_summary.csv        one row per holdout (pooled + mean±SD)
  <out_dir>/per_unit/<holdout>.csv       per-protein / per-cluster / per-category
  <out_dir>/predictions/<holdout>.parquet out-of-fold preds (for scatter plots)

The suite maps 1:1 to docs/benchmark_plan.md.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ddg.evaluation import splits
from ddg.evaluation.labels import add_label_columns
from ddg.evaluation.metrics import compute_metrics, summarize
from ddg.evaluation.models import build_xy, make_model

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResults:
    summary: pd.DataFrame                       # one row per holdout
    per_unit: dict = field(default_factory=dict)     # holdout -> DataFrame
    predictions: dict = field(default_factory=dict)  # holdout -> DataFrame(y,pred,unit)


# ----- fold execution -----

def _fit_predict(model_name, X, y, tr, te):
    model = make_model(model_name)
    model.fit(X[tr], y[tr])
    return model.predict(X[te])


def _run_cv(df, X, y, model_name, kind, unit_col, k, seed):
    """CV holdout: collect out-of-fold preds for every row, score per-unit."""
    oof = np.full(len(df), np.nan)
    covered = np.zeros(len(df), dtype=bool)
    for _, tr, te in splits.cv_folds(df, kind if kind != "random" else "random",
                                     k=k, seed=seed):
        oof[te] = _fit_predict(model_name, X, y, tr, te)
        covered[te] = True
    if not covered.any():
        return None, None, None

    pred_df = pd.DataFrame({
        "y": y[covered], "pred": oof[covered],
        "unit": df[unit_col].to_numpy()[covered] if unit_col else np.nan,
    })
    pooled = compute_metrics(pred_df["y"], pred_df["pred"])

    per_unit_rows, per_unit_metrics = [], []
    if unit_col:
        for unit, g in pred_df.groupby("unit"):
            m = compute_metrics(g["y"], g["pred"])
            m_row = {"unit": unit, **m}
            per_unit_rows.append(m_row)
            per_unit_metrics.append(m)
    per_unit_df = pd.DataFrame(per_unit_rows) if per_unit_rows else None
    dist = summarize(per_unit_metrics) if per_unit_metrics else {}
    return pooled, dist, (per_unit_df, pred_df)


def _run_leaveout(df, X, y, model_name, col, gen, **gen_kw):
    """Leave-category-out holdout: one test category per fold, train on the rest."""
    rows, preds = [], []
    for label, tr, te in gen(df, col, **gen_kw):
        pred = _fit_predict(model_name, X, y, tr, te)
        m = compute_metrics(y[te], pred)
        rows.append({"unit": label, **m})
        preds.append(pd.DataFrame({"y": y[te], "pred": pred, "unit": label}))
    if not rows:
        return None, None, None
    per_unit_df = pd.DataFrame(rows)
    pred_df = pd.concat(preds, ignore_index=True)
    pooled = compute_metrics(pred_df["y"], pred_df["pred"])
    dist = summarize(rows)
    return pooled, dist, (per_unit_df, pred_df)


# ----- the suite -----

def run_benchmark(
    features: pd.DataFrame,
    model_name: str = "svr",
    out_dir: str | Path | None = None,
    cluster_map: dict | None = None,
    k_random: int = 10,
    k_group: int = 5,
    seed: int = 42,
    holdouts: list[str] | None = None,
    drop_s: bool = False,
) -> BenchmarkResults:
    """
    Run holdouts on a features table (must have wt_id, mutation, ddg + features).

    holdouts: subset of {'random','protein','cluster','denovo','substitution',
    'source_residue','target_residue','chemistry'}; default = all feasible.
    drop_s: exclude s-derived features (the s-ablation).
    """
    df = add_label_columns(features, cluster_map=cluster_map).reset_index(drop=True)
    X, y, feat_cols = build_xy(df, drop_s=drop_s)
    logger.info("Benchmark on %d mutations, %d features, model=%s, drop_s=%s",
                len(df), len(feat_cols), model_name, drop_s)

    all_holdouts = ["random", "protein", "cluster", "denovo", "substitution",
                    "source_residue", "target_residue", "chemistry"]
    holdouts = holdouts or all_holdouts

    summary_rows, per_unit, predictions = [], {}, {}

    def record(name, pooled, dist, extra):
        if pooled is None:
            logger.warning("holdout '%s' produced no folds; skipped", name)
            return
        row = {"holdout": name, "model": model_name,
               "n": pooled["n"], "pooled_pearson": pooled["pearson"],
               "pooled_spearman": pooled["spearman"],
               "pooled_rmse": pooled["rmse"], "pooled_mae": pooled["mae"]}
        row.update(dist or {})
        summary_rows.append(row)
        if extra:
            per_unit_df, pred_df = extra
            if per_unit_df is not None:
                per_unit[name] = per_unit_df
            predictions[name] = pred_df
        logger.info("  %-15s pooled r=%.3f  n=%d  units=%s",
                    name, pooled["pearson"], pooled["n"],
                    (dist or {}).get("n_units", "-"))

    if "random" in holdouts:
        record("random", *_run_cv(df, X, y, model_name, "random", None, k_random, seed))
    if "protein" in holdouts:
        record("protein", *_run_cv(df, X, y, model_name, "protein", "protein", k_group, seed))
    if "cluster" in holdouts:
        if "cluster" in df.columns and df["cluster"].notna().any():
            record("cluster", *_run_cv(df, X, y, model_name, "cluster", "cluster", k_group, seed))
        else:
            logger.warning("cluster holdout skipped: no cluster_map provided "
                           "(see ddg.evaluation.cluster to build one)")
    if "denovo" in holdouts:
        record("denovo", *_run_leaveout(df, X, y, model_name, "class_natural",
                                        splits.binary_transfer_folds))
    if "substitution" in holdouts:
        record("substitution", *_run_leaveout(df, X, y, model_name, "substitution",
                                               splits.leave_out_folds))
    if "source_residue" in holdouts:
        record("source_residue", *_run_leaveout(df, X, y, model_name, "wt_aa",
                                                 splits.leave_out_folds))
    if "target_residue" in holdouts:
        record("target_residue", *_run_leaveout(df, X, y, model_name, "mut_aa",
                                                 splits.leave_out_folds))
    if "chemistry" in holdouts:
        record("chemistry", *_run_leaveout(df, X, y, model_name, "chem_category",
                                           splits.leave_out_folds))

    summary = pd.DataFrame(summary_rows)
    results = BenchmarkResults(summary=summary, per_unit=per_unit, predictions=predictions)

    if out_dir is not None:
        _write(results, out_dir)
    return results


def _write_table(df, path_no_ext):
    """Write parquet when an engine is available, else fall back to CSV."""
    try:
        df.to_parquet(f"{path_no_ext}.parquet", index=False)
    except (ImportError, ValueError):
        df.to_csv(f"{path_no_ext}.csv", index=False)


def _write(results: BenchmarkResults, out_dir):
    out = Path(out_dir)
    (out / "per_unit").mkdir(parents=True, exist_ok=True)
    (out / "predictions").mkdir(parents=True, exist_ok=True)
    results.summary.to_csv(out / "benchmark_summary.csv", index=False)
    for name, dfu in results.per_unit.items():
        dfu.to_csv(out / "per_unit" / f"{name}.csv", index=False)
    for name, dfp in results.predictions.items():
        _write_table(dfp, out / "predictions" / name)
    logger.info("Wrote benchmark tables to %s", out)
