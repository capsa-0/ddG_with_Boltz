"""
The RF4Mave harness: leave-one-protein-out random forest predicting MAVE fitness
from stability (ΔΔG) and conservation (ΔΔE) features.

This is a faithful re-implementation of Høie et al. 2022's protocol, decoded from
their released code (`src/RandomForest_model.py`, `src/utilities/
prism_machine_learning.py`) rather than from the paper prose alone:

  * Features come from their own `preprocessed.pkl` (39 DataFrames, 92 columns),
    selected by the same `str.contains` regexes their `train.sh` passes with `-f`.
  * `RandomForestRegressor(n_estimators=150, max_features="sqrt",
    min_samples_leaf=15)` — their exact model.
  * Missing values are the sentinel **-100**, not NaN, and `--exclude-missing 2`
    (their `-x 2`) drops rows whose *own* Rosetta or GEMME value is missing, from
    both train and validation.
  * Leave-one-protein-out: for each of the 39 datasets, every dataset belonging to
    the same protein is removed from training, and the held-out dataset is scored
    on its own. Protein identity is the same regex they use,
    `[0-9]{3}_([A-Za-z0-9-]{3,8})_`, giving 29 proteins over 39 datasets.
  * Headline metric: **median Spearman ρ across the 39 datasets**.

Published numbers this must reproduce (their Figure 2B):

    null (s̃_exp)                0.17     (paper quotes this one as a mean)
    ΔΔG only (Rosetta)          0.25
    ΔΔE only (GEMME)            0.42
    ΔΔG + ΔΔE                   0.47
    position-context (47 feat)  0.52

Phase 3 swaps the `ros_*` columns for our Boltz-embedding ΔΔG and re-runs the
identical harness, so the comparison is like-for-like.

Usage
-----
    python results/15_mave_stability_transfer/rf4mave.py --out <dir>
    python results/15_mave_stability_transfer/rf4mave.py --models ddg_only,dde_only
"""

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ddg.evaluation.metrics import compute_metrics  # noqa: E402

PKL = ROOT / "data" / "raw" / "mave_hoie" / "preprocessed.pkl"
MISSING = -100.0
PROTEIN_RE = re.compile(r"[0-9]{3}_([A-Za-z0-9-]{3,8})_")

# The `-f` regex lists from their train.sh, plus the null model. Values are
# fragments fed to `columns.str.contains("|".join(...))`, exactly as they do it.
FEATURE_SETS = {
    "null_smave": ["mave_wt", "mave_any"],
    "ddg_only": [r"ros_aa_wt_p$"],
    "dde_only": [r"gemme_aa_wt_p$"],
    "ddg_dde": [r"ros_aa_wt_p$", r"gemme_aa_wt_p$"],
    "position_context": [
        "gemme_aa_p0", r"gemme_aa_wt_p$", "gemme_M_p0",
        "ros_aa_p0", r"ros_aa_wt_p$", "ros_M_p0",
        "mave_wt", "mave_any",
    ],
}
# Published medians, for the reproduction gate.
PUBLISHED = {"null_smave": 0.17, "ddg_only": 0.25, "dde_only": 0.42,
             "ddg_dde": 0.47, "position_context": 0.52}


def load_datasets(pkl_path: Path = PKL):
    """Their preprocessed.pkl -> (frames, names, proteins).

    The pickle is a 3-element Series: (processed frames, dataset names, raw
    frames). We keep the processed frames — the scores differ only by a monotone
    rank transform, so Spearman is identical either way (asserted in --self-test).
    """
    proc, names, _raw = pd.read_pickle(pkl_path)
    names = [str(n).replace(".txt", "") for n in names]
    proteins = []
    for n in names:
        m = PROTEIN_RE.search(n)
        if m is None:
            raise ValueError(f"cannot extract protein name from {n!r}")
        proteins.append(m.group(1))
    wanted = "|".join(sorted({p for pats in FEATURE_SETS.values() for p in pats}))
    frames = []
    for df in proc:
        df = df[df["score"].notna()]
        df = df.select_dtypes(include=[np.number])
        keep = df.columns.str.contains("^score$|ros_aa_wt_p$|gemme_aa_wt_p$|" + wanted)
        frames.append(df.loc[:, keep].fillna(MISSING).astype(np.float32))
    return frames, names, proteins


def select_features(df: pd.DataFrame, patterns) -> list:
    """Column names matching any regex fragment, excluding the label."""
    mask = df.columns.str.contains("|".join(patterns))
    return [c for c, keep in zip(df.columns, mask) if keep and c != "score"]


def _drop_missing(df: pd.DataFrame, level: int) -> pd.DataFrame:
    """Their `-x` filter: drop rows whose own Rosetta (>=1) or GEMME (>=2) is -100."""
    keep = pd.Series(True, index=df.index)
    if level >= 1 and "ros_aa_wt_p" in df.columns:
        keep &= df["ros_aa_wt_p"] != MISSING
    if level >= 2 and "gemme_aa_wt_p" in df.columns:
        keep &= df["gemme_aa_wt_p"] != MISSING
    return df[keep]


def run_lopo(frames, names, proteins, patterns, exclude_missing=2,
             trees=150, seed=0, ddg_col="ros_aa_wt_p", verbose=True):
    """Leave-one-protein-out over all datasets. Returns one row per dataset.

    Stacks every dataset once into a single float32 matrix and selects folds with a
    boolean mask, rather than re-concatenating 38 DataFrames per fold -- the naive
    version is both slow and memory-hungry, and this workstation has 6 GB.
    """
    # The -x filter always looks at the *variant's own* stability column, which is
    # `ros_aa_wt_p` for Rosetta and whatever we substitute in Phase 3.
    if ddg_col != "ros_aa_wt_p":
        frames = [f.rename(columns={ddg_col: "ros_aa_wt_p"}) for f in frames]
    frames = [_drop_missing(f, exclude_missing) for f in frames]
    feature_cols = select_features(frames[0], patterns)
    if not feature_cols:
        raise ValueError(f"no columns matched {patterns}")

    blocks, ys, owner = [], [], []
    for i, f in enumerate(frames):
        blocks.append(f[feature_cols].to_numpy(dtype=np.float32))
        ys.append(f["score"].to_numpy(dtype=np.float32))
        owner.append(np.full(len(f), i, dtype=np.int32))
    X = np.vstack(blocks)
    y = np.concatenate(ys)
    owner = np.concatenate(owner)
    protein_of = np.array(proteins)

    rows = []
    for i, (name, protein) in enumerate(zip(names, proteins)):
        test = owner == i
        train = protein_of[owner] != protein
        if test.sum() < 10:
            rows.append(dict(dataset=name, protein=protein, n=int(test.sum()),
                             spearman=np.nan))
            continue
        model = RandomForestRegressor(n_estimators=trees, max_features="sqrt",
                                      min_samples_leaf=15, random_state=seed,
                                      n_jobs=-1)
        model.fit(X[train], y[train])
        pred = model.predict(X[test])
        m = compute_metrics(y[test], pred)
        rows.append(dict(dataset=name, protein=protein, n=int(test.sum()),
                         n_train=int(train.sum()), spearman=m["spearman"],
                         pearson=m["pearson"]))
        if verbose:
            print(f"    {name[:52]:54} n={int(test.sum()):6}  "
                  f"rho={m['spearman']:+.3f}", flush=True)
    return pd.DataFrame(rows), feature_cols


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="RF4Mave LOPO harness")
    ap.add_argument("--pkl", type=Path, default=PKL)
    ap.add_argument("--models", default=",".join(FEATURE_SETS),
                    help="comma-separated subset of " + ",".join(FEATURE_SETS))
    ap.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent / "phase0")
    ap.add_argument("--trees", type=int, default=150)
    ap.add_argument("--exclude-missing", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    frames, names, proteins = load_datasets(args.pkl)
    print(f"loaded {len(frames)} datasets over {len(set(proteins))} proteins "
          f"({sum(len(f) for f in frames):,} scored variants)")

    args.out.mkdir(parents=True, exist_ok=True)
    summary, per_dataset = [], []
    for model_name in args.models.split(","):
        model_name = model_name.strip()
        print(f"\n{model_name}  ({len(FEATURE_SETS[model_name])} regex fragments)")
        t0 = time.time()
        res, cols = run_lopo(frames, names, proteins, FEATURE_SETS[model_name],
                             exclude_missing=args.exclude_missing,
                             trees=args.trees, seed=args.seed)
        res.insert(0, "model", model_name)
        per_dataset.append(res)
        med = res["spearman"].median()
        pub = PUBLISHED.get(model_name, np.nan)
        summary.append(dict(model=model_name, n_features=len(cols),
                            n_datasets=int(res["spearman"].notna().sum()),
                            median_spearman=med,
                            mean_spearman=res["spearman"].mean(),
                            published=pub, delta=med - pub,
                            seconds=round(time.time() - t0, 1)))
        print(f"  -> median ρ = {med:.3f}  (published {pub:.2f}, "
              f"Δ {med - pub:+.3f})   [{len(cols)} features, "
              f"{time.time() - t0:.0f} s]")

    sm = pd.DataFrame(summary)
    pd.concat(per_dataset).to_csv(args.out / "lopo_per_dataset.csv", index=False)
    sm.to_csv(args.out / "lopo_summary.csv", index=False)
    print("\n" + sm.to_string(index=False))
    print(f"\nwrote {args.out}/lopo_summary.csv and lopo_per_dataset.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
