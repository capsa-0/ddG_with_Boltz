"""EXP 1 — predict the per-protein offset from a WHOLE-PROTEIN Boltz representation.

Earlier null used the mean of wtz over MUTATED POSITIONS only (median 3 sites on S669).
Here the descriptor is the full WT single representation s (L,384) pooled over ALL
residues (mean+sd = 768 dims), on Tsuboyama, where ~30 mutations/protein make the
offset a genuine protein property rather than a curation artefact.

Also reports the honest (split-half) oracle ceiling on Tsuboyama.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCR = ROOT / "data/processed/_analysis"
SCR.mkdir(parents=True, exist_ok=True)
AAS = "ACDEFGHIKLMNPQRSTVWY"


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-9 else np.nan


def rho(a, b):
    return r(pd.Series(a).rank(), pd.Series(b).rank())


def cv_pred(X, y, groups, model="ridge", n_splits=5):
    pred = np.full(len(y), np.nan)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        imp = SimpleImputer(strategy="median").fit(X[tr])
        sca = StandardScaler().fit(imp.transform(X[tr]))
        T = lambda Q: sca.transform(imp.transform(Q))
        if model == "ridge":
            m = RidgeCV(alphas=np.logspace(-2, 5, 30)).fit(T(X[tr]), y[tr])
            pred[te] = m.predict(T(X[te]))
        else:
            ms = [MLPRegressor((256, 64), alpha=1e-2, max_iter=800, early_stopping=True,
                               random_state=s).fit(T(X[tr]), y[tr]) for s in range(3)]
            pred[te] = np.mean([m.predict(T(X[te])) for m in ms], axis=0)
    return pred


def main():
    df = pd.read_csv(SCR / "tsu_mut_classes.csv")
    ids = [l.strip() for l in open(SCR / "tsu_s_ids.txt")]
    S = np.load(SCR / "tsu_s_desc.npy")
    clus = pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/cluster_map_30.csv") \
        .set_index("protein_id").cluster
    seqs = pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/mutations.csv",
                       usecols=["wt_id", "sequence_wt"]).groupby("wt_id").sequence_wt.first()

    df = df[df.wt_id.isin(ids)].copy()
    n_per = df.groupby("wt_id").size()
    print(f"Tsuboyama held-out (OOF): {len(df)} muts / {df.wt_id.nunique()} proteins")
    print(f"variants per protein: median {n_per.median():.0f}, mean {n_per.mean():.1f}")
    print(f"OOF overall: r={r(df.y, df.pred_OOF):.3f} rho={rho(df.y, df.pred_OOF):.3f}\n")

    # ---------- how much would a perfect offset buy, honestly? ----------
    ok = n_per[n_per >= 6].index
    sub = df[df.wt_id.isin(ok)].copy()
    rng = np.random.default_rng(0)
    rel, base, hon, orc = [], [], [], []
    for _ in range(100):
        hA, hB = [], []
        for p, s in sub.groupby("wt_id"):
            idx = rng.permutation(s.index.to_numpy())
            k = len(idx) // 2
            hA.append(idx[:k]); hB.append(idx[k:])
        A, B = sub.loc[np.concatenate(hA)], sub.loc[np.concatenate(hB)]
        rel.append(r(A.groupby("wt_id").y.mean().loc[B.groupby("wt_id").y.mean().index],
                     B.groupby("wt_id").y.mean()))
        offA = (A.y - A.pred_OOF).groupby(A.wt_id).mean()
        offB = (B.y - B.pred_OOF).groupby(B.wt_id).mean()
        base.append(r(B.y, B.pred_OOF))
        hon.append(r(B.y, B.pred_OOF + B.wt_id.map(offA).to_numpy()))
        orc.append(r(B.y, B.pred_OOF + B.wt_id.map(offB).to_numpy()))
    f = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    print(f"=== honest offset ceiling on Tsuboyama ({len(ok)} proteins >=6 muts) ===")
    print(f"  split-half reliability of per-protein mean ddG : {f(rel)}")
    print(f"  baseline (no offset)                           : {f(base)}")
    print(f"  offset from the OTHER half [honest]            : {f(hon)}")
    print(f"  offset from the SAME half  [in-sample]         : {f(orc)}")
    print(f"  -> real transferable gain {np.mean(hon)-np.mean(base):+.3f}\n")

    # Persist: this is the in-distribution half of the experiment's headline contrast
    # and it previously existed only as printed prose.
    HERE = Path(__file__).resolve().parent
    pd.DataFrame([
        dict(quantity="split_half_reliability_mean_ddg", mean=np.mean(rel), sd=np.std(rel)),
        dict(quantity="baseline_no_offset", mean=np.mean(base), sd=np.std(base)),
        dict(quantity="offset_from_other_half_honest", mean=np.mean(hon), sd=np.std(hon)),
        dict(quantity="offset_from_same_half_oracle", mean=np.mean(orc), sd=np.std(orc)),
        dict(quantity="transferable_gain", mean=np.mean(hon) - np.mean(base), sd=float("nan")),
    ]).to_csv(HERE / "split_half_tsuboyama.csv", index=False)
    print(f"wrote {HERE / 'split_half_tsuboyama.csv'}")

    # ---------- can the whole-protein s predict it? ----------
    off = (df.y - df.pred_OOF).groupby(df.wt_id).mean()
    keep = [i for i in ids if i in off.index]
    idx = [ids.index(i) for i in keep]
    yv = off.loc[keep].to_numpy(float)
    Sk = S[idx]
    groups = clus.loc[keep].to_numpy()
    seq = seqs.loc[keep]
    comp = np.array([[s.count(a) / len(s) for a in AAS] for s in seq])
    length = np.log(np.array([len(s) for s in seq], float))[:, None]

    print("=== predicting the offset, 5-fold GroupKFold on 30% homology cluster ===")
    print(f"  target sd = {yv.std():.2f} kcal/mol, n = {len(yv)} proteins, "
          f"{len(np.unique(groups))} clusters\n")
    const = yv.std()
    for tag, X, mdl in (
            ("length only", length, "ridge"),
            ("length + composition", np.hstack([length, comp]), "ridge"),
            ("Boltz whole-protein s (ridge)", Sk, "ridge"),
            ("Boltz whole-protein s (MLP)", Sk, "mlp"),
            ("s mean-pool only", Sk[:, :384], "ridge"),
            ("s + length + composition", np.hstack([Sk, length, comp]), "ridge")):
        pr = cv_pred(X, yv, groups, mdl)
        m = dict(zip(keep, pr))
        corr = df.pred_OOF + df.wt_id.map(m).to_numpy()
        print(f"  {tag:31s} offset r={r(yv, pr):+.3f} RMSE={np.sqrt(np.mean((yv-pr)**2)):.2f} "
              f"(const {const:.2f}) | pooled r {r(df.y, df.pred_OOF):.3f} -> {r(df.y, corr):.3f}")

    # for reference: dG(WT) as the offset predictor, if it were known perfectly
    dg = pd.read_csv(SCR / "tsu_dG_wt.csv").set_index("wt_id").dG_wt
    common = [i for i in keep if i in dg.index]
    print(f"\n  [reference] true dG(WT) vs the offset, n={len(common)}: "
          f"r={r(dg.loc[common].to_numpy(), off.loc[common].to_numpy()):+.3f}")


if __name__ == "__main__":
    main()
