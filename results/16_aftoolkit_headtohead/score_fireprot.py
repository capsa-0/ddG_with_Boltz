"""Score AFToolkit on FireProt and compare it to this project, paired.

Consumes the per-variant 358-d AF2 feature vectors produced on the cluster by
`aft_fireprot_run.py`, runs AFToolkit's released SVM adapter on them, and reports the
comparison split by whether AFToolkit has already trained on that protein.

    rsync -a cranex:/grupos/.../aftoolkit/features/ $AFT/fireprot_features/
    AFT=... python results/16_aftoolkit_headtohead/score_fireprot.py
"""
import os
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
AFT = Path(os.environ.get("AFT", "/tmp/aftoolkit"))
FEATDIR = Path(os.environ.get("FEATDIR", AFT / "fireprot_features"))
B, SEED = 4000, 0
OURS = ["diag", "dz_cw", "dz", "cw", "base"]


def load_ours():
    d = None
    for f in ("exp14_fpfilt_results_locality_paired.csv", "exp14_fpfilt_results_onehot_fp.csv",
              "exp14_fpfilt_results_farctrl.csv", "exp14_fpfilt_results_fact_noaug.csv"):
        x = pd.read_csv(ROOT / "data/processed/_analysis" / f).reset_index(drop=True)
        if d is None:
            d = x.copy()
            continue
        assert d[["wt_id", "mutation", "ddg"]].equals(x[["wt_id", "mutation", "ddg"]])
        for c in x.columns:
            if c not in d:
                d[c] = x[c].values
    return d


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=len(y), r=pearsonr(y, p)[0], rho=spearmanr(y, p).statistic,
                rmse=float(np.sqrt(np.mean((y - p) ** 2))), mae=float(np.mean(np.abs(y - p))))


def paired_bootstrap(df, a, b):
    rng = np.random.default_rng(SEED)
    prots = df.wt_id.unique()
    idx = {p: df.index[df.wt_id == p].to_numpy() for p in prots}
    y, pa, pb = df.ddg.to_numpy(), df[a].to_numpy(), df[b].to_numpy()
    dr, drho = [], []
    for _ in range(B):
        ii = np.concatenate([idx[p] for p in rng.choice(prots, len(prots), replace=True)])
        if y[ii].std() == 0:
            continue
        dr.append(pearsonr(y[ii], pa[ii])[0] - pearsonr(y[ii], pb[ii])[0])
        drho.append(spearmanr(y[ii], pa[ii]).statistic - spearmanr(y[ii], pb[ii]).statistic)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(d_r=float(np.mean(dr)), d_r_lo=q(dr)[0], d_r_hi=q(dr)[1],
                p_r_gt0=float(np.mean(np.array(dr) > 0)),
                d_rho=float(np.mean(drho)), d_rho_lo=q(drho)[0], d_rho_hi=q(drho)[1],
                p_rho_gt0=float(np.mean(np.array(drho) > 0)))


def main():
    sys.path.insert(0, str(AFT / "shim"))
    task = pd.read_csv(AFT / "fireprot_aft_task.csv")
    have = {f.stem for f in FEATDIR.glob("*.npy")}
    missing = set(task.aft_id) - have
    if missing:
        print(f"[warn] {len(missing)} of {len(task)} variants have no AF2 features yet "
              f"(extraction still running?); scoring what is present")
    task = task[task.aft_id.isin(have)]
    X = np.stack([np.load(FEATDIR / f"{i}.npy") for i in task.aft_id])
    model = pickle.load(open(AFT / "trained_svm.pkl", "rb"))
    task["aft_svm"] = model.base_model.predict(X)

    m = load_ours().merge(task[["wt_id", "mutation", "aft_svm"]], on=["wt_id", "mutation"])
    ov = pd.read_csv(R / "fireprot_aftoolkit_train_overlap.csv")
    m = m.merge(ov[["wt_id", "mutation", "protein_in_aft_train"]].drop_duplicates(),
                on=["wt_id", "mutation"], how="left")
    m = m.reset_index(drop=True)
    m.to_csv(R / "fireprot_paired_predictions.csv", index=False)

    rows, diffs = [], []
    for tag, d in (("all scored", m),
                   ("AFToolkit HAS trained on these proteins", m[m.protein_in_aft_train]),
                   ("blind to both methods", m[~m.protein_in_aft_train])):
        d = d.reset_index(drop=True)
        print(f"\n=== FireProt — {tag}: {len(d)} variants / {d.wt_id.nunique()} proteins")
        sub = [dict(subset=tag, model="AFToolkit SVM", **metrics(d.ddg, d.aft_svm))]
        sub += [dict(subset=tag, model=f"ours: {c}", **metrics(d.ddg, d[c])) for c in OURS if c in d]
        print(pd.DataFrame(sub).round(3).to_string(index=False))
        rows += sub
        if d.wt_id.nunique() > 2:
            for c in ("diag", "dz_cw"):
                diffs.append(dict(subset=tag, ours=c, **paired_bootstrap(d, c, "aft_svm")))
                x = diffs[-1]
                print(f"  paired Δ (ours {c} − AFToolkit): "
                      f"ρ {x['d_rho']:+.3f} [{x['d_rho_lo']:+.3f}, {x['d_rho_hi']:+.3f}] "
                      f"P={x['p_rho_gt0']:.3f} | r {x['d_r']:+.3f} "
                      f"[{x['d_r_lo']:+.3f}, {x['d_r_hi']:+.3f}] P={x['p_r_gt0']:.3f}")
    pd.DataFrame(rows).to_csv(R / "headtohead_fireprot.csv", index=False)
    pd.DataFrame(diffs).to_csv(R / "headtohead_fireprot_bootstrap.csv", index=False)
    print(f"\nwrote {R/'headtohead_fireprot.csv'}, {R/'headtohead_fireprot_bootstrap.csv'}, "
          f"{R/'fireprot_paired_predictions.csv'}")


if __name__ == "__main__":
    main()
