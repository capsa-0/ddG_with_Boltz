"""
Module: splits
Description: Train/test fold generators for each holdout.

Two families:
  * CV-style (``cv_folds``): every row is in exactly one test fold, so the
    benchmark can collect out-of-fold predictions for the whole dataset and then
    slice per-unit distributions. Used for random / protein / cluster.
  * leave-out (``leave_out_folds``): each unique value of a label is one test
    fold, trained on everything else. Used for substitution / source residue /
    target residue / chemistry / de-novo.

Each generator yields (fold_label, train_idx, test_idx) with numpy int indices
into the (row-reset) dataframe.
"""

import numpy as np
from sklearn.model_selection import GroupKFold, KFold


def cv_folds(df, kind: str, k: int = 5, seed: int = 42):
    """
    Yield CV folds.

    kind='random' -> plain KFold over rows.
    kind='<col>'  -> GroupKFold keeping each group's rows entirely in one fold
                     (e.g. 'protein', 'cluster').
    """
    n = len(df)
    idx = np.arange(n)
    if kind == "random":
        splitter = KFold(n_splits=k, shuffle=True, random_state=seed)
        for i, (tr, te) in enumerate(splitter.split(idx)):
            yield f"fold{i}", tr, te
        return

    if kind not in df.columns:
        raise ValueError(f"group column '{kind}' not in dataframe")
    groups = df[kind].to_numpy()
    valid = np.array([g is not None and g == g for g in groups])  # drop NaN groups
    n_groups = len(np.unique(groups[valid]))
    k_eff = min(k, n_groups)
    if k_eff < 2:
        return
    splitter = GroupKFold(n_splits=k_eff)
    sub = idx[valid]
    for i, (tr, te) in enumerate(splitter.split(sub, groups=groups[valid])):
        yield f"fold{i}", sub[tr], sub[te]


def leave_out_folds(df, col: str, min_test: int = 10, min_train: int = 50):
    """
    Yield one fold per unique value of ``col``: test = rows with that value,
    train = all other rows. Skips values with too few test or train points.
    """
    if col not in df.columns:
        raise ValueError(f"column '{col}' not in dataframe")
    idx = np.arange(len(df))
    values = df[col]
    for val in sorted(v for v in values.dropna().unique()):
        test_mask = (values == val).to_numpy()
        te = idx[test_mask]
        tr = idx[~test_mask]
        if te.size < min_test or tr.size < min_train:
            continue
        yield str(val), tr, te


def binary_transfer_folds(df, col: str):
    """
    Yield both directions of a two-class transfer split (e.g. natural<->designed):
    (train=A, test=B) and (train=B, test=A). ``col`` must have exactly 2 classes.
    """
    idx = np.arange(len(df))
    classes = [c for c in df[col].dropna().unique()]
    if len(classes) != 2:
        return
    a, b = classes
    a_mask = (df[col] == a).to_numpy()
    yield f"train_{a}__test_{b}", idx[a_mask], idx[~a_mask]
    yield f"train_{b}__test_{a}", idx[~a_mask], idx[a_mask]
