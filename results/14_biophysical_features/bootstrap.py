"""14 — Are the feature-block differences real, or protein-sampling noise?

Cluster bootstrap over PROTEINS (the unit of independence: mutations inside one
protein share a wild-type structure, one embedding and one assay batch, so their
errors are correlated and a per-mutation bootstrap would give intervals that are far
too narrow). Resample the 412 proteins with replacement, take every mutation of each
drawn protein, recompute each metric, repeat N_BOOT times.

Reported as the **paired difference vs a reference configuration**, computed on the
same resample so the shared protein draw cancels — the same protocol as
`results/13_balanced_loss/bootstrap.py`.

Runs on saved out-of-fold predictions; no model is refit.

    python results/14_biophysical_features/bootstrap.py \
        --oof exp14_oof_results_cons.csv --ref base
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
STAB = -0.5
N_BOOT = 400
KEYS = ["r", "rho", "mae", "stab_rho", "stab_bias", "auc_stab", "detpr30", "ndcg30"]


def ndcg_stab(y, pred, k=30):
    gain = np.maximum(0.0, -y)
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gain[np.argsort(pred)[:k]] * disc).sum())
    ideal = float((np.sort(gain)[::-1][:k] * disc).sum())
    return dcg / ideal if ideal > 0 else np.nan


def stats(y, p):
    stab = y < STAB
    return {
        "r": np.corrcoef(y, p)[0, 1],
        "rho": spearmanr(y, p).statistic,
        "mae": np.abs(p - y).mean(),
        "stab_rho": spearmanr(y[stab], p[stab]).statistic if stab.sum() > 5 else np.nan,
        "stab_bias": (p[stab] - y[stab]).mean() if stab.any() else np.nan,
        "auc_stab": roc_auc_score(stab, -p) if 0 < stab.sum() < len(y) else np.nan,
        "detpr30": stab[np.argsort(p)[:30]].mean(),
        "ndcg30": ndcg_stab(y, p),
    }


def ci(vals):
    v = np.asarray(vals, float)
    v = v[np.isfinite(v)]
    if v.size < 20:
        return np.nan, np.nan, np.nan
    return float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default="exp14_oof_results_cons.csv")
    ap.add_argument("--ref", default="base", help="reference config for the paired diff")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    d = pd.read_csv(ROOT / "data/processed/_analysis" / args.oof).reset_index(drop=True)
    cfgs = [c for c in d.columns if c not in ("wt_id", "mutation", "ddg")]
    assert args.ref in cfgs, f"reference '{args.ref}' not in {cfgs}"
    others = [c for c in cfgs if c != args.ref]
    prots = d.wt_id.unique()
    by = {p: g.index.to_numpy() for p, g in d.groupby("wt_id")}
    rng = np.random.default_rng(0)

    boot = {c: {m: [] for m in KEYS} for c in cfgs}
    diff = {c: {m: [] for m in KEYS} for c in others}
    for _ in range(args.boot):
        pick = rng.choice(prots, len(prots), replace=True)
        idx = np.concatenate([by[p] for p in pick])
        s = d.iloc[idx]
        y = s.ddg.to_numpy(float)
        if (y < STAB).sum() < 10:            # degenerate draw: no tail to score
            continue
        cur = {c: stats(y, s[c].to_numpy(float)) for c in cfgs}
        for c in cfgs:
            for m in KEYS:
                boot[c][m].append(cur[c][m])
        for c in others:
            for m in KEYS:
                diff[c][m].append(cur[c][m] - cur[args.ref][m])

    n_ok = len(boot[args.ref]["r"])
    print(f"cluster bootstrap over {len(prots)} proteins, {n_ok}/{args.boot} usable "
          f"resamples; reference = {args.ref}\n")

    rows = []
    print(f"=== paired difference vs {args.ref}  (95% CI; * excludes zero) ===")
    print(f"{'metric':11s}" + "".join(f"{c:>26s}" for c in others))
    for m in KEYS:
        line = f"{m:11s}"
        for c in others:
            mean, lo, hi = ci(diff[c][m])
            star = "*" if np.isfinite(lo) and (lo > 0 or hi < 0) else " "
            line += f"{mean:+8.3f} [{lo:+.3f},{hi:+.3f}]{star}".rjust(26)
            rows.append({"config": c, "metric": m, "kind": "paired_diff",
                         "mean": mean, "lo": lo, "hi": hi,
                         "significant": bool(star.strip())})
        print(line)

    print(f"\n=== absolute metric [95% CI] ===")
    for m in KEYS:
        line = f"{m:11s}"
        for c in cfgs:
            mean, lo, hi = ci(boot[c][m])
            line += f"{mean:7.3f} [{lo:.3f},{hi:.3f}]".rjust(24)
            rows.append({"config": c, "metric": m, "kind": "absolute",
                         "mean": mean, "lo": lo, "hi": hi, "significant": None})
        print(line)

    out = OUT / f"bootstrap_{Path(args.oof).stem}_ref-{args.ref.replace('+','-')}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
