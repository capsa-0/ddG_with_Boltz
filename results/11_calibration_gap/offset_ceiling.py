"""How much could a per-protein correction buy on S669?

Oracle decomposition: take the existing regime A/B/D predictions and apply the BEST
POSSIBLE per-protein correction (offset / gain / affine), then re-score. That is the
ceiling for any 'predict a per-protein number from the WT embedding' scheme.
Then test whether the required offset is learnable from the WT embedding at all.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
sys.path.insert(0, str(ROOT))

Z, N_SEED = 128, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]
WT = [f"wtz_{j}" for j in range(Z)]
OUT = ROOT / "results/09_external_benchmarks"


def mat(df):
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def augment(X, y):
    Xa = np.concatenate([X[:, Z:], X[:, :Z]], axis=1)
    return np.vstack([X, Xa]), np.concatenate([y, -y])


def members():
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=250, early_stopping=False,
                         random_state=s, warm_start=True) for s in range(N_SEED)]


def predict(ms, T, X):
    return np.mean([m.predict(T(X)) for m in ms], axis=0)


def r(a, b):
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def per_prot_median(y, p, prot):
    v = [r(y[prot == q], p[prot == q]) for q in np.unique(prot)]
    v = [x for x in v if np.isfinite(x)]
    return float(np.median(v)) if v else np.nan


# ---------- oracle corrections ----------
def oracle(y, pred, prot, mode):
    out = pred.copy().astype(float)
    for q in np.unique(prot):
        m = prot == q
        if m.sum() < 3:
            continue
        yt, yp = y[m], pred[m]
        if mode == "offset":                       # shift only
            out[m] = yp + (yt.mean() - yp.mean())
        elif mode == "gain":                       # rescale about own mean only
            s = yp.std()
            out[m] = yp.mean() + (yp - yp.mean()) * (yt.std() / s if s > 1e-9 else 1.0)
        elif mode == "affine":                     # shift + rescale
            s = yp.std()
            out[m] = yt.mean() + (yp - yp.mean()) * (yt.std() / s if s > 1e-9 else 1.0)
    return out


def main():
    tsu = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
    fp = pd.read_parquet(ROOT / "data/processed/fireprot_le500/features_ablation.parquet")
    bdf = pd.read_parquet(ROOT / "data/processed/s669/features_ablation.parquet")
    leak = pd.read_csv(OUT / "homology" / "s669_leakage.csv").set_index("protein")

    Xt, yt = mat(tsu), tsu["ddg"].to_numpy(float)
    Xta, yta = augment(Xt, yt)
    impA = SimpleImputer(strategy="median").fit(Xta)
    scaA = StandardScaler().fit(impA.transform(Xta))
    TA = lambda X: scaA.transform(impA.transform(X))
    A = members()
    for m in A:
        m.fit(TA(Xta), yta)

    Xf, yf = mat(fp), fp["ddg"].to_numpy(float)
    Xfa, yfa = augment(Xf, yf)
    impB = SimpleImputer(strategy="median").fit(Xfa)
    scaB = StandardScaler().fit(impB.transform(Xfa))
    TB = lambda X: scaB.transform(impB.transform(X))
    B = members()
    for m in B:
        m.fit(TB(Xfa), yfa)

    D = copy.deepcopy(A)
    for m in D:
        m.set_params(learning_rate_init=1e-3, max_iter=400)
        m.fit(TA(Xfa), yfa)

    y = bdf["ddg"].to_numpy(float)
    prot = bdf["wt_id"].to_numpy()
    X = mat(bdf)

    # ---- variance decomposition of the TRUTH ----
    dfy = pd.DataFrame({"y": y, "p": prot})
    gm = dfy.groupby("p")["y"].transform("mean")
    vb, vw = float(np.var(gm)), float(np.var(y - gm))
    print(f"S669 truth: n={len(y)} / {len(np.unique(prot))} proteins")
    print(f"  ddG variance  between-protein {vb:.3f} ({vb/(vb+vw):.1%})   "
          f"within-protein {vw:.3f} ({vw/(vb+vw):.1%})")
    print(f"  per-protein mean ddG: mean {gm.mean():.2f}  sd {np.std(np.unique(gm)):.2f} kcal/mol\n")

    rows = []
    subsets = {"full": np.ones(len(y), bool),
               "common25": np.array([not bool(leak.loc[p, "leaky_any_25"])
                                     if p in leak.index else True for p in prot])}

    for name, ms, T in (("A_tsu_only", A, TA), ("B_fp_only", B, TB), ("D_finetuned", D, TA)):
        p0 = predict(ms, T, X)
        if r(y, p0) < 0:
            p0 = -p0                       # same sign convention as run_benchmarks
        for sub, keep in subsets.items():
            yk, pk, prk = y[keep], p0[keep], prot[keep]
            base = r(yk, pk)
            row = {"regime": name, "subset": sub, "n": int(keep.sum()),
                   "baseline_r": base, "baseline_rmse": rmse(yk, pk),
                   "per_prot_med_r": per_prot_median(yk, pk, prk)}
            for mode in ("offset", "gain", "affine"):
                c = oracle(yk, pk, prk, mode)
                row[f"oracle_{mode}_r"] = r(yk, c)
                row[f"oracle_{mode}_rmse"] = rmse(yk, c)
            rows.append(row)

    res = pd.DataFrame(rows)
    print("=== ORACLE CEILING on S669 (pooled Pearson r) ===")
    print(res[["regime", "subset", "n", "baseline_r", "oracle_offset_r",
               "oracle_gain_r", "oracle_affine_r", "per_prot_med_r"]]
          .round(3).to_string(index=False))
    print("\n=== same, RMSE kcal/mol ===")
    print(res[["regime", "subset", "baseline_rmse", "oracle_offset_rmse",
               "oracle_gain_rmse", "oracle_affine_rmse"]].round(3).to_string(index=False))

    # ---- is the required offset learnable from the WT embedding? ----
    print("\n=== IS THE OFFSET PREDICTABLE FROM THE WT EMBEDDING? ===")
    for name, ms, T in (("B_fp_only", B, TB), ("D_finetuned", D, TA)):
        p0 = predict(ms, T, X)
        if r(y, p0) < 0:
            p0 = -p0
        g = pd.DataFrame({"p": prot, "res": y - p0})
        need = g.groupby("p")["res"].mean()                    # required offset per protein
        desc = bdf.groupby("wt_id")[WT].mean().loc[need.index]  # protein-level WT descriptor
        Xd = np.nan_to_num(desc.to_numpy(float))
        yd = need.to_numpy(float)

        # leave-one-protein-out ridge, INSIDE S669 (optimistic: same benchmark)
        pr = np.empty_like(yd)
        for i in range(len(yd)):
            m = np.ones(len(yd), bool)
            m[i] = False
            mdl = RidgeCV(alphas=np.logspace(-2, 4, 25)).fit(Xd[m], yd[m])
            pr[i] = mdl.predict(Xd[i:i + 1])[0]
        print(f"  {name}: required offset sd = {yd.std():.2f} kcal/mol; "
              f"LOPO-predicted r = {r(yd, pr):.3f}, "
              f"RMSE {rmse(yd, pr):.2f} vs predict-the-mean {rmse(yd, np.full_like(yd, yd.mean())):.2f}")

        # what would applying that LOPO-predicted offset do to pooled r?
        off = dict(zip(need.index, pr))
        corrected = p0 + np.array([off[q] for q in prot])
        print(f"           applying it: pooled r {r(y, p0):.3f} -> {r(y, corrected):.3f}, "
              f"RMSE {rmse(y, p0):.2f} -> {rmse(y, corrected):.2f}")

    res.to_csv(str(SCR / "offset_ceiling.csv"), index=False)


if __name__ == "__main__":
    main()
