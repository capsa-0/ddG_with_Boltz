"""
07_feature_symmetry_ablation — do concat features and/or symmetry augmentation help?

Within-dataset protein-holdout (GroupKFold by wt_id) on Tsuboyama and FireProt
separately, out-of-fold pooled metrics, for the 2x2 ablation:
  features:   'dz'     = zdiag_* + zpool_*   (current pipeline, the WT-mut DIFFERENCE)
              'concat' = wtz_* + mtz_*        (old notebook, pooled WT and mutant levels)
  augment:    'none' | 'sym'  (symmetry: add reversed mutation with negated ddg to TRAIN only)

Symmetry reversal:
  dz     -> features negate, ddg negate          (Δz(B->A) = -Δz(A->B))
  concat -> swap [wtz|mtz] halves, ddg negate     (WT<->mut)

Model = the project MLP (5-seed ensemble). Prints a table and writes results.csv.

    python results/07_feature_symmetry_ablation/run_ablation.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ddg.evaluation.models import make_model
from ddg.evaluation.metrics import compute_metrics

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
OUT = ROOT / "results/07_feature_symmetry_ablation"
DATASETS = {
    "Tsuboyama": ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet",
    "FireProt": ROOT / "data/processed/fireprot_le500/features_ablation.parquet",
}
FEATSETS = {"dz": ["zdiag", "zpool"], "concat": ["wtz", "mtz"]}
K, Z = 5, 128


def cols(prefixes):
    return [f"{p}_{j}" for p in prefixes for j in range(Z)]


def augment(X, y, kind):
    """Return the symmetry-reversed (X, y) for the given feature kind."""
    if kind == "dz":
        return -X, -y
    # concat: swap the two 128-d halves
    return np.concatenate([X[:, Z:], X[:, :Z]], axis=1), -y


def oof(df, feat_cols, kind, aug):
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    y = df["ddg"].to_numpy(float)
    groups = df["wt_id"].to_numpy()
    pred = np.full(len(df), np.nan)
    for tr, te in GroupKFold(n_splits=K).split(X, y, groups):
        Xtr, ytr = X[tr], y[tr]
        if aug == "sym":
            Xa, ya = augment(Xtr, ytr, kind)
            Xtr, ytr = np.vstack([Xtr, Xa]), np.concatenate([ytr, ya])
        m = make_model("mlp")
        m.fit(Xtr, ytr)
        pred[te] = m.predict(X[te])
    return compute_metrics(y, pred)


rows = []
for ds, path in DATASETS.items():
    df = pd.read_parquet(path)
    for kind, prefixes in FEATSETS.items():
        fc = cols(prefixes)
        for aug in ("none", "sym"):
            m = oof(df, fc, kind, aug)
            rows.append({"dataset": ds, "features": kind, "augment": aug,
                         "n": m["n"], "pearson": m["pearson"],
                         "spearman": m["spearman"], "rmse": m["rmse"], "mae": m["mae"]})
            print(f"{ds:10s} {kind:6s} {aug:4s}  r={m['pearson']:.3f} "
                  f"rho={m['spearman']:.3f} RMSE={m['rmse']:.3f}  n={m['n']}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
print("\n=== summary (pooled Pearson r, protein-holdout) ===")
print(res.pivot_table(index=["dataset", "features"], columns="augment",
                      values="pearson").round(3).to_string())
print(f"\nwrote {OUT/'results.csv'}")
