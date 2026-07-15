"""
Module: pipeline
Description: The ordered pipeline steps, their run functions, and filesystem-based
progress detectors used by `ddg status`.

Run functions are imported lazily so that lightweight commands (status/list) and
cheap steps don't drag in heavy or cluster-only dependencies (umap, boltz).
"""

import logging
from pathlib import Path

import pandas as pd

from ddg.config.config_loader import ProjectConfig

logger = logging.getLogger(__name__)

# Steps that this CLI can currently run locally (predict shells out to boltz).
RUNNABLE = ["prepare", "predict", "slim", "features"]


def slim_dir(config: ProjectConfig) -> Path:
    return config.exp_processed_dir / "slim"


# ----- Run functions (lazy imports inside) -----

def _run_prepare(exp_cfg, names_cfg):
    from ddg.feature_extraction.generate_queries import main as generate_queries
    generate_queries(exp_cfg, names_cfg)


def _run_predict(exp_cfg, names_cfg, shard=None):
    from ddg.feature_extraction.extract_features import main as extract_features
    extract_features(exp_cfg, names_cfg, shard=shard)


def _run_slim(exp_cfg, names_cfg):
    from ddg.storage.slim import positions_by_structure, slim_predictions
    config = ProjectConfig(exp_cfg, names_cfg)
    df = pd.read_csv(config.mutations_df_path)
    pos_by_struct = positions_by_structure(df)
    predictions = Path(config.raw_features_dir) / "predictions"
    out = slim_dir(config) / "shard_0000.npz"
    keep_s = bool(config.exp_config.get("slim", {}).get("keep_s", True))
    delete_raw = bool(config.exp_config.get("slim", {}).get("delete_raw", False))
    slim_predictions(predictions, pos_by_struct, out,
                     keep_s=keep_s, delete_raw=delete_raw)


def _run_features(exp_cfg, names_cfg):
    from ddg.exploration.explore_features import main as explore_features
    explore_features(exp_cfg, names_cfg)


RUN_FUNCS = {
    "prepare": _run_prepare,
    "predict": _run_predict,
    "slim": _run_slim,
    "features": _run_features,
}


# ----- Progress detectors (done, expected|None, unit) -----

def _expected_counts(config: ProjectConfig):
    p = config.mutations_df_path
    if not Path(p).exists():
        return None
    df = pd.read_csv(p)
    n_mut = len(df)
    key_col = "wt_key" if "wt_key" in df.columns else "wt_id"
    n_wt = int(df[key_col].nunique())
    return {"n_mut": n_mut, "n_wt": n_wt, "n_struct": n_wt + n_mut}


def _count(path: Path, pattern: str) -> int:
    path = Path(path)
    return len(list(path.glob(pattern))) if path.exists() else 0


def progress(step: str, config: ProjectConfig):
    """Return (done, expected_or_None, unit) inferred from the filesystem."""
    exp = _expected_counts(config)
    n_struct = exp["n_struct"] if exp else None

    if step == "prepare":
        done = exp["n_mut"] if exp else 0
        return done, (exp["n_mut"] if exp else None), "mutations"
    if step == "msa":
        return _count(config.msa_dir, "*.a3m"), n_struct, "a3m"
    if step == "queries":
        return _count(config.queries_dir, "*.yaml"), n_struct, "yaml"
    if step == "predict":
        preds = Path(config.raw_features_dir) / "predictions"
        return _count(preds, "*/embeddings_*.npz"), n_struct, "npz"
    if step == "slim":
        done = 0
        for f in Path(slim_dir(config)).glob("*.npz"):
            try:
                import numpy as np
                with np.load(f, allow_pickle=False) as d:
                    done += len(d["keys"])
            except Exception:
                pass
        return done, n_struct, "slim"
    if step == "features":
        parquet = config.exp_processed_dir / "features_summary.parquet"
        if Path(parquet).exists():
            try:
                return len(pd.read_parquet(parquet)), (exp["n_mut"] if exp else None), "rows"
            except Exception:
                return 1, None, "rows"
        return 0, (exp["n_mut"] if exp else None), "rows"
    return 0, None, ""
