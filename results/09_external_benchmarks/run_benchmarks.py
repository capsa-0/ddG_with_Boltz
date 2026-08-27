"""
09_external_benchmarks — score the ΔΔG predictor on the S669 and Ssym blind benchmarks
under three training regimes, each reported full and homology-filtered.

Regimes (all: concat wtz+mtz features, antisymmetry augmentation, 5-seed MLP ensemble;
identical machinery to results/08 run_finetune.py):
  A. Tsuboyama-only : fit on ALL Tsuboyama.
  B. FireProt-only  : fit on ALL FireProt (own imputer/scaler).
  D. Fine-tuned     : pretrain on Tsuboyama, warm-start continue on FireProt.

For each benchmark x regime we report pooled r/rho/RMSE/MAE + per-protein median, on:
  - full     : every benchmark variant.
  - filt<thr>: drop benchmark proteins homologous (>= thr) to the regime's training
               corpus (A->leaky_tsu, B->leaky_fp, D->leaky_any), from build_homology_map.py.
Ssym additionally reports antisymmetry: the reverse mutation's features are the direct
features with wtz/mtz halves swapped, so pred_reverse = model(swap(X)); we report the
direct-vs-(-reverse) correlation and the antisymmetry bias mean(pred_dir + pred_rev).

Sign convention: benchmark ddG may be signed oppositely; if pooled Pearson < 0 we flip
the prediction sign (recorded) before reporting RMSE/MAE, as ddg.evaluation.transfer does.

    conda run -n ddG_with_Boltz python results/09_external_benchmarks/run_benchmarks.py

Requires the benchmark features (built on the cluster):
    data/processed/{s669,ssym}/features_ablation.parquet
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ddg.evaluation.metrics import compute_metrics  # noqa: E402

OUT = Path(__file__).resolve().parent
Z, N_SEED = 128, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]
THRESHOLDS = (25, 30)

TSU_PATH = ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet"
FP_PATH = ROOT / "data/processed/fireprot_le500/features_ablation.parquet"
BENCH = {
    "s669": ROOT / "data/processed/s669/features_ablation.parquet",
    "ssym": ROOT / "data/processed/ssym/features_ablation.parquet",
}
LEAK_COL = {"A_tsu_only": "leaky_tsu", "B_fp_only": "leaky_fp", "D_finetuned": "leaky_any"}


def mat(df):
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def augment(X, y):
    """Antisymmetry for concat: swap [wtz|mtz] halves, negate ddg."""
    Xa = np.concatenate([X[:, Z:], X[:, :Z]], axis=1)
    return np.vstack([X, Xa]), np.concatenate([y, -y])


def members():
    """The project-default estimator (as in ddg.evaluation.models.make_model('mlp')).

    CORRECTED 2026-08-27. This previously used `max_iter=250, early_stopping=False`,
    which overfits: `max_iter` counts EPOCHS, so regime A (24,718 augmented samples)
    took ~4x the gradient updates of regime B (6,410) and was penalised hardest —
    confounding the very A-vs-B-vs-D comparison this study makes. Isolating the change
    on S669 raised regime A from r 0.255 to 0.415. `warm_start` is retained because
    regime D fine-tunes from regime A's fitted weights.
    """
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=1000, early_stopping=True,
                         n_iter_no_change=25, validation_fraction=0.1,
                         random_state=s, warm_start=True) for s in range(N_SEED)]


def predict(ms, T, X):
    return np.mean([m.predict(T(X)) for m in ms], axis=0)


def score(y, pred):
    """compute_metrics, flipping prediction sign if pooled Pearson < 0."""
    m = compute_metrics(y, pred)
    flip = np.isfinite(m["pearson"]) and m["pearson"] < 0
    if flip:
        m = compute_metrics(y, -pred)
    m["sign_flipped"] = bool(flip)
    return m, (-pred if flip else pred)


def per_protein_median(y, pred, prot):
    rs = []
    for p in np.unique(prot):
        mm = compute_metrics(y[prot == p], pred[prot == p])
        if np.isfinite(mm["pearson"]):
            rs.append(mm["pearson"])
    return (float(np.median(rs)) if rs else np.nan), len(rs)


def main():
    tsu = pd.read_parquet(TSU_PATH)
    fp = pd.read_parquet(FP_PATH)
    print(f"train: Tsuboyama {len(tsu)} muts / {tsu.wt_id.nunique()} prot; "
          f"FireProt {len(fp)} muts / {fp.wt_id.nunique()} prot")

    # ---- train the three regimes once ----
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

    D = copy.deepcopy(A)  # fine-tune: warm-start A on FireProt, reuse Tsuboyama transform
    for m in D:
        m.set_params(learning_rate_init=1e-3, max_iter=400)  # early stopping still applies
        m.fit(TA(Xfa), yfa)

    regimes = [("A_tsu_only", A, TA), ("B_fp_only", B, TB), ("D_finetuned", D, TA)]

    rows, anti_rows = [], []
    for bench, path in BENCH.items():
        if not path.exists():
            print(f"[skip] {bench}: {path} not found (run cluster feature extraction first)")
            continue
        bdf = pd.read_parquet(path)
        leak = pd.read_csv(OUT / "homology" / f"{bench}_leakage.csv").set_index("protein")
        y = bdf["ddg"].to_numpy(float)
        prot = bdf["wt_id"].to_numpy()
        X = mat(bdf)
        print(f"\n=== {bench}: {len(bdf)} variants / {bdf.wt_id.nunique()} proteins ===")

        for cond, ms, T in regimes:
            pred0 = predict(ms, T, X)
            mfull, pred = score(y, pred0)
            pm, npm = per_protein_median(y, pred, prot)
            rows.append({"benchmark": bench, "regime": cond, "subset": "full",
                         "n_prot": bdf.wt_id.nunique(), "per_prot_median_r": pm,
                         **mfull})
            print(f"  {cond:11s} full     r={mfull['pearson']:.3f} rho={mfull['spearman']:.3f}"
                  f" RMSE={mfull['rmse']:.3f} n={mfull['n']} (per-prot med r={pm:.3f})"
                  f"{' [sign-flipped]' if mfull['sign_flipped'] else ''}")
            # Two filtered views per threshold:
            #  filt{thr}   : drop proteins homologous to THIS regime's training corpus.
            #  common{thr} : drop proteins homologous to ANY training corpus (leaky_any),
            #                identical mask for all regimes -> a fair cross-regime subset.
            for thr in THRESHOLDS:
                for tag, col in ((f"filt{thr}", f"{LEAK_COL[cond]}_{thr}"),
                                 (f"common{thr}", f"leaky_any_{thr}")):
                    keep = np.array([not bool(leak.loc[p, col]) if p in leak.index else True
                                     for p in prot])
                    mf, pf = score(y[keep], pred0[keep])
                    pmf, _ = per_protein_median(y[keep], pf, prot[keep])
                    rows.append({"benchmark": bench, "regime": cond, "subset": tag,
                                 "n_prot": int(len(np.unique(prot[keep]))),
                                 "per_prot_median_r": pmf, **mf})
                    print(f"  {cond:11s} {tag:9s} r={mf['pearson']:.3f} rho={mf['spearman']:.3f}"
                          f" RMSE={mf['rmse']:.3f} n={mf['n']} (per-prot med r={pmf:.3f})")

            if bench == "ssym":  # antisymmetry from the direct features
                Xrev = np.concatenate([X[:, Z:], X[:, :Z]], axis=1)
                pr = predict(ms, T, Xrev)
                am = compute_metrics(pred0, -pr)
                anti_rows.append({"regime": cond, "antisym_r": am["pearson"],
                                  "bias_mean": float(np.mean(pred0 + pr)),
                                  "bias_sd": float(np.std(pred0 + pr))})
                print(f"  {cond:11s} antisymmetry: corr(dir,-rev)={am['pearson']:.3f} "
                      f"bias={np.mean(pred0 + pr):+.3f}±{np.std(pred0 + pr):.3f}")

    res = pd.DataFrame(rows)
    res.to_csv(OUT / "results.csv", index=False)
    if anti_rows:
        pd.DataFrame(anti_rows).to_csv(OUT / "ssym_antisymmetry.csv", index=False)
    print("\n=== pooled Pearson (rows=benchmark/regime, cols=subset) ===")
    if not res.empty:
        print(res.pivot_table(index=["benchmark", "regime"], columns="subset",
                              values="pearson").round(3).to_string())
    print(f"\nwrote {OUT / 'results.csv'}")


if __name__ == "__main__":
    main()
