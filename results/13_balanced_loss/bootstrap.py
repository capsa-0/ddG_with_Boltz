"""Are the balanced-loss differences real, or fold/sampling noise?

Cluster bootstrap over PROTEINS (the unit of independence) on the saved OOF
predictions, giving CIs on each metric and — more informative — on the paired
difference BMC − MSE, which cancels the shared resample.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAB = -0.5
N_BOOT = 400


def stats(y, p):
    stab = y < STAB
    out = {"rho": spearmanr(y, p).statistic,
           "r": np.corrcoef(y, p)[0, 1],
           "mae": np.abs(p - y).mean(),
           "stab_bias": (p[stab] - y[stab]).mean() if stab.sum() else np.nan,
           "detpr30": stab[np.argsort(p)[:30]].mean()}
    out["auc_stab"] = (roc_auc_score(stab, -p) if 0 < stab.sum() < len(y) else np.nan)
    out["stab_rho"] = (spearmanr(y[stab], p[stab]).statistic if stab.sum() > 5 else np.nan)
    return out


def main():
    d = pd.read_csv(ROOT / "data/processed/_analysis/balanced_oof.csv")
    prots = d.wt_id.unique()
    by = {p: g.index.to_numpy() for p, g in d.groupby("wt_id")}
    rng = np.random.default_rng(0)
    keys = ["rho", "r", "mae", "stab_rho", "stab_bias", "auc_stab", "detpr30"]

    boot = {k: {m: [] for m in keys} for k in ("mse", "bmc", "lds")}
    diff = {k: {m: [] for m in keys} for k in ("bmc", "lds")}

    for _ in range(N_BOOT):
        pick = rng.choice(prots, len(prots), replace=True)
        idx = np.concatenate([by[p] for p in pick])
        s = d.iloc[idx]
        y = s.y.to_numpy()
        if (y < STAB).sum() < 10:
            continue
        cur = {k: stats(y, s[f"pred_{k}"].to_numpy()) for k in ("mse", "bmc", "lds")}
        for k in cur:
            for m in keys:
                boot[k][m].append(cur[k][m])
        for k in ("bmc", "lds"):
            for m in keys:
                diff[k][m].append(cur[k][m] - cur["mse"][m])

    print(f"cluster bootstrap over {len(prots)} proteins, {N_BOOT} resamples\n")
    print("=== metric [95% CI] ===")
    hdr = f"{'metric':11s}" + "".join(f"{k:>22s}" for k in ("mse", "bmc", "lds"))
    print(hdr)
    for m in keys:
        row = f"{m:11s}"
        for k in ("mse", "bmc", "lds"):
            v = np.array(boot[k][m], float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            row += f"{np.nanmean(v):8.3f} [{lo:5.2f},{hi:5.2f}]"
        print(row)

    print("\n=== paired difference vs MSE (95% CI; * = excludes 0) ===")
    for k in ("bmc", "lds"):
        print(f"  --- {k} − mse ---")
        for m in keys:
            v = np.array(diff[k][m], float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            sig = "*" if (lo > 0) or (hi < 0) else " "
            print(f"    {m:11s} {np.nanmean(v):+7.3f} [{lo:+.3f}, {hi:+.3f}] {sig}")

    rows = []
    for k in ("mse", "bmc", "lds"):
        for m in keys:
            v = np.array(boot[k][m], float)
            lo, hi = np.nanpercentile(v, [2.5, 97.5])
            rows.append({"loss": k, "metric": m, "mean": np.nanmean(v),
                         "lo95": lo, "hi95": hi})
    pd.DataFrame(rows).to_csv(OUT / "bootstrap.csv", index=False)
    print(f"\nwrote {OUT/'bootstrap.csv'}")


if __name__ == "__main__":
    main()
