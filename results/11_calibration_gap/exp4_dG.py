"""EXP 4 — can Boltz trunk embeddings predict absolute dG(WT)?

Target : dG_ML of the WT construct (Tsuboyama 2023 supplementary), per protein.
Features: whole-protein pooled single representation s (L,384) -> mean+std (768),
          from the WT structure's slim entry. Baselines: length, composition.
CV     : GroupKFold on the homology cluster (data/raw/tsuboyama_bench_clusters.csv).

If this fails, the 'predict a protein-level calibration term' line is dead.
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


def build_s_descriptors(slim_dir, wanted):
    """mean and sd over residues of the WT single representation."""
    out = {}
    for f in sorted(Path(slim_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        keys = [str(k) for k in d["keys"]]
        for i, k in enumerate(keys):
            if k not in wanted or k in out:
                continue
            s = d[f"s_{i}"]
            if s.ndim == 3:
                s = s[-1]
            s = s.astype(np.float32)
            out[k] = np.concatenate([s.mean(0), s.std(0)])
    return out


def cv_eval(X, y, groups, tag, model="ridge", n_splits=5):
    pred = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=n_splits)
    for tr, te in gkf.split(X, y, groups):
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
    const = np.sqrt(np.mean((y - y.mean()) ** 2))
    print(f"  {tag:34s} r={r(y, pred):+.3f}  rho={rho(y, pred):+.3f}  "
          f"RMSE={np.sqrt(np.mean((y - pred) ** 2)):.2f} (const {const:.2f})")
    return pred


def main():
    # ---- target: dG of the WT construct ----
    wt = pd.read_csv(SCR / "tsu_wt_dG.csv")
    wt["dG"] = pd.to_numeric(wt.dG_ML, errors="coerce")
    dG = wt.dropna(subset=["dG"]).groupby("WT_name").dG.median()

    mut = pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/mutations.csv",
                      usecols=["wt_id", "sequence_wt"])
    seqs = mut.groupby("wt_id").sequence_wt.first()
    clus = pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/cluster_map_30.csv").set_index("protein_id").cluster

    desc = build_s_descriptors(ROOT / "data/processed/tsuboyama_bench_fast/slim",
                               set(seqs.index))
    ids = sorted(set(desc) & set(dG.index) & set(clus.index))
    print(f"proteins with WT embedding + dG + cluster: {len(ids)}")

    y = dG.loc[ids].to_numpy(float)
    groups = clus.loc[ids].to_numpy()
    S = np.vstack([desc[i] for i in ids])
    seq = seqs.loc[ids]
    comp = np.array([[s.count(a) / len(s) for a in AAS] for s in seq])
    length = np.log(np.array([len(s) for s in seq], float))[:, None]

    print(f"dG(WT): mean {y.mean():.2f}  sd {y.std():.2f} kcal/mol  "
          f"range [{y.min():.2f}, {y.max():.2f}]")
    print(f"clusters: {len(np.unique(groups))}\n")

    print("=== predicting dG(WT), 5-fold GroupKFold on homology cluster ===")
    cv_eval(length, y, groups, "length only (ridge)")
    cv_eval(comp, y, groups, "composition 20aa (ridge)")
    cv_eval(np.hstack([length, comp]), y, groups, "length + composition (ridge)")
    cv_eval(S, y, groups, "Boltz whole-protein s (ridge)")
    cv_eval(S, y, groups, "Boltz whole-protein s (MLP)", model="mlp")
    cv_eval(np.hstack([S, length, comp]), y, groups, "s + length + composition (ridge)")

    # split s into mean-only / sd-only to see which half carries it
    half = S.shape[1] // 2
    cv_eval(S[:, :half], y, groups, "  s mean-pool only (ridge)")
    cv_eval(S[:, half:], y, groups, "  s sd-over-residues only (ridge)")

    pd.DataFrame({"wt_id": ids, "dG_wt": y, "cluster": groups}).to_csv(
        SCR / "tsu_dG_wt.csv", index=False)
    np.save(SCR / "tsu_s_desc.npy", S)
    with open(SCR / "tsu_s_ids.txt", "w") as fh:
        fh.write("\n".join(ids))
    print(f"\nsaved descriptors ({S.shape}) for reuse in exp 1")


if __name__ == "__main__":
    main()
