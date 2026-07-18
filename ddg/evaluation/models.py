"""
Module: models
Description: Regressor factory + feature-matrix helper for the benchmark.

Every model is an sklearn Pipeline: median-impute -> standardize -> estimator, so
NaN/inf-cleaned features and unscaled inputs are handled identically across
holdouts. SVR (RBF) is the project's working model; Ridge is a fast linear
baseline; MLP is optional.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, VotingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from ddg.evaluation.labels import feature_columns


def make_model(name: str = "hgb") -> Pipeline:
    """
    Return a fresh pipeline for the named estimator.

    'hgb'   — HistGradientBoosting: fast, strong on tabular, scales to the wide
              corpus (SVR is O(n^2-3) and infeasible past ~10k samples/fold).
    'svr'   — RBF SVR (the project's original model; only tractable on small folds).
    'ridge' — fast linear baseline.
    'mlp'   — seed-ensembled MLP (see below).
    """
    name = name.lower()
    if name == "hgb":
        est = HistGradientBoostingRegressor(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
            l2_regularization=1.0, early_stopping=True, random_state=0)
    elif name == "svr":
        est = SVR(kernel="rbf", C=10.0, gamma="scale", epsilon=0.1)
    elif name == "ridge":
        est = Ridge(alpha=1.0)
    elif name == "mlp":
        # A single MLP is high-variance on this tabular problem — its group-holdout
        # score depends on the weight init and the internal early-stopping split.
        # Average several seed-decorrelated nets (the standard variance-reduction
        # fix), each a moderately deep, L2-regularized MLP that early-stops
        # patiently on its own validation split. n_jobs fits the members in
        # parallel, so the ensemble costs ~one net of wall time.
        def _member(seed):
            return MLPRegressor(
                hidden_layer_sizes=(256, 128, 64), activation="relu",
                solver="adam", alpha=3e-3, learning_rate_init=1e-3,
                batch_size=256, max_iter=1000, early_stopping=True,
                n_iter_no_change=25, validation_fraction=0.1, random_state=seed)
        est = VotingRegressor(
            estimators=[(f"mlp{s}", _member(s)) for s in range(5)], n_jobs=-1)
    else:
        raise ValueError(f"unknown model '{name}' (use hgb | svr | ridge | mlp)")
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("est", est),
    ])


def build_xy(df: pd.DataFrame, feat_cols: list[str] | None = None,
             drop_s: bool = False):
    """
    Extract (X, y, feat_cols) from a labelled features table.

    Infs are converted to NaN so the pipeline's imputer can handle them; columns
    that are entirely non-finite are dropped. drop_s excludes s-derived columns.
    """
    if feat_cols is None:
        feat_cols = feature_columns(df, drop_s=drop_s)
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan)
    keep = [c for c in feat_cols if X[c].notna().any()]
    X = X[keep].to_numpy(dtype=float)
    y = df["ddg"].to_numpy(dtype=float)
    return X, y, keep
