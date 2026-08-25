"""Is the per-protein offset gain REAL, or was it fitting sampling noise?

split-half oracle: estimate each protein's offset from HALF its variants (true labels),
apply it to the OTHER half, score only on that half. If the gain survives, the offset
is a stable protein-level quantity worth predicting. If it collapses, the in-sample
oracle (+0.24) was noise-fitting.

Also: can interpretable structure/chemistry descriptors predict it better than the
Boltz embedding did? (length, composition, burial from Boltz distogram, hydropathy)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCR = ROOT / "data/processed/_analysis"
SCR.mkdir(parents=True, exist_ok=True)
AAS = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-9 else np.nan


def main():
    df = pd.read_csv(SCR / "s669_predictions.csv")
    df["pred"] = df["pred_D"]
    n_per = df.groupby("wt_id").size()
    print(f"S669: {len(df)} variants / {df.wt_id.nunique()} proteins")
    print(f"variants per protein: median {n_per.median():.0f}, "
          f"mean {n_per.mean():.1f}, >=6: {(n_per >= 6).sum()} proteins "
          f"({n_per[n_per >= 6].sum()} variants)\n")

    # ---------- 1. split-half reliability of the per-protein mean ddG ----------
    ok = n_per[n_per >= 6].index
    sub = df[df.wt_id.isin(ok)].copy()
    rng = np.random.default_rng(0)
    rel, orc, hon, base = [], [], [], []
    for rep in range(200):
        hA, hB = [], []
        for p, s in sub.groupby("wt_id"):
            idx = rng.permutation(s.index.to_numpy())
            k = len(idx) // 2
            hA.append(idx[:k]); hB.append(idx[k:])
        A = sub.loc[np.concatenate(hA)]
        B = sub.loc[np.concatenate(hB)]
        mA = A.groupby("wt_id").y.mean()
        mB = B.groupby("wt_id").y.mean()
        rel.append(r(mA.loc[mB.index], mB))

        offA = (A.y - A.pred).groupby(A.wt_id).mean()      # offset from half A only
        offB = (B.y - B.pred).groupby(B.wt_id).mean()      # in-sample offset for B
        base.append(r(B.y, B.pred))
        hon.append(r(B.y, B.pred + B.wt_id.map(offA).to_numpy()))
        orc.append(r(B.y, B.pred + B.wt_id.map(offB).to_numpy()))

    f = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    print("=== SPLIT-HALF, scored on held-out half (proteins with >=6 variants) ===")
    print(f"  split-half reliability of per-protein mean ddG : {f(rel)}")
    print(f"  baseline (no offset)                           : {f(base)}")
    print(f"  offset from the OTHER half  [honest]           : {f(hon)}")
    print(f"  offset from the SAME half   [in-sample oracle] : {f(orc)}")
    print(f"  -> real transferable gain  {np.mean(hon) - np.mean(base):+.3f}"
          f"   |  noise-fitting inflation {np.mean(orc) - np.mean(hon):+.3f}\n")

    # ---------- 2. interpretable descriptors ----------
    mut = pd.read_csv(ROOT / "data/processed/s669/mutations.csv")
    seqs = mut.groupby("wt_id").sequence_wt.first()
    bur = pd.read_csv(SCR / "s669_mut_classes.csv")[["wt_id", "burial"]]
    bstat = bur.groupby("wt_id").burial.agg(["mean", "std", "max"]).add_prefix("bur_")

    rows = []
    for p, s in seqs.items():
        c = {f"f_{a}": s.count(a) / len(s) for a in AAS}
        c.update(wt_id=p, length=len(s),
                 kd_mean=np.mean([KD.get(x, 0) for x in s]),
                 f_hydroph=sum(s.count(a) for a in "AVLIMFWC") / len(s),
                 f_charged=sum(s.count(a) for a in "DEKR") / len(s),
                 f_GP=sum(s.count(a) for a in "GP") / len(s))
        rows.append(c)
    D = pd.DataFrame(rows).set_index("wt_id").join(bstat)
    D["log_len"] = np.log(D.length)

    need = (df.y - df.pred).groupby(df.wt_id).mean()
    D = D.loc[need.index].fillna(D.median(numeric_only=True))
    yv = need.to_numpy(float)

    print("=== CAN INTERPRETABLE DESCRIPTORS PREDICT THE OFFSET? (LOPO ridge) ===")
    print(f"  target: per-protein offset, sd = {yv.std():.2f} kcal/mol, n = {len(yv)} proteins\n")
    sets = {
        "length only":            ["log_len"],
        "burial only":            ["bur_mean", "bur_std", "bur_max"],
        "composition (20 aa)":    [f"f_{a}" for a in AAS],
        "chemistry summary":      ["kd_mean", "f_hydroph", "f_charged", "f_GP"],
        "ALL descriptors":        ["log_len", "bur_mean", "bur_std", "bur_max", "kd_mean",
                                   "f_hydroph", "f_charged", "f_GP"] + [f"f_{a}" for a in AAS],
    }
    for tag, cols in sets.items():
        Xd = D[cols].to_numpy(float)
        pr = np.empty(len(yv))
        for i in range(len(yv)):
            m = np.ones(len(yv), bool); m[i] = False
            sc = StandardScaler().fit(Xd[m])
            mdl = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(sc.transform(Xd[m]), yv[m])
            pr[i] = mdl.predict(sc.transform(Xd[i:i + 1]))[0]
        off = dict(zip(D.index, pr))
        corr = df.pred + df.wt_id.map(off).to_numpy()
        print(f"  {tag:22s} offset r={r(yv, pr):+.3f}  "
              f"RMSE {np.sqrt(np.mean((yv-pr)**2)):.2f} (const {yv.std():.2f})  "
              f"| pooled r {r(df.y, df.pred):.3f} -> {r(df.y, corr):.3f}")

    # simple univariate correlations, for intuition
    print("\n  univariate |r| of each descriptor with the offset:")
    uni = {c: r(D[c].to_numpy(float), yv) for c in
           ["log_len", "bur_mean", "bur_std", "bur_max", "kd_mean", "f_hydroph",
            "f_charged", "f_GP"]}
    for k, v in sorted(uni.items(), key=lambda kv: -abs(kv[1])):
        print(f"    {k:12s} {v:+.3f}")


if __name__ == "__main__":
    main()
