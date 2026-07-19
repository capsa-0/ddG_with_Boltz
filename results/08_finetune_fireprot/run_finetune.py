"""
08_finetune_fireprot — sequentially fine-tune a Tsuboyama-pretrained MLP on FireProt,
test on BOTH datasets, under the cross-dataset homology splits (30/50/90 %).

Uses the project defaults adopted in 07: **concat features** (wtz+mtz) and
**antisymmetry augmentation** (concat: swap the two halves + negate ΔΔG) on every
training set.

Conditions (per identity threshold):
  A. Tsuboyama-only : pretrain on tsu_train, predict tsu_test & fp_test
  D. Fine-tuned     : pretrain on tsu_train, warm-start continue on fp_finetune,
                      predict tsu_test & fp_test
The pretrain is shared: A uses it directly, D deep-copies it and continues on FireProt.
Imputer/scaler are fit on the (augmented) tsu_train and reused for the FireProt stage,
so the model's input space stays fixed across fine-tuning.

    python results/08_finetune_fireprot/run_finetune.py
writes results.csv and prints a table.
"""
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ddg.evaluation.metrics import compute_metrics

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
OUT = ROOT / "results/08_finetune_fireprot"
Z, N_SEED = 128, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]   # concat features
TSU = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
FP = pd.read_parquet(ROOT / "data/processed/fireprot_le200/features_ablation.parquet")


def xy(df, ids):
    d = df[df.wt_id.isin(ids)]
    X = d[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)
    return X, d["ddg"].to_numpy(float)


def augment(X, y):
    """Antisymmetry for concat: swap [wtz|mtz] halves, negate ddg; append to (X,y)."""
    Xa = np.concatenate([X[:, Z:], X[:, :Z]], axis=1)
    return np.vstack([X, Xa]), np.concatenate([y, -y])


def members():
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=250, early_stopping=False,
                         random_state=s, warm_start=True) for s in range(N_SEED)]


def predict(ms, T, X):
    return np.mean([m.predict(T(X)) for m in ms], axis=0)


rows = []
for NN in (30, 50, 90):
    split = pd.read_csv(OUT / f"splits/split_{NN}.csv")
    S = {s: set(split[split.set == s].protein_id) for s in split.set.unique()}
    Xtr, ytr = xy(TSU, S["tsu_train"])
    Xa, ya = augment(Xtr, ytr)
    imp = SimpleImputer(strategy="median").fit(Xa)
    sca = StandardScaler().fit(imp.transform(Xa))
    T = lambda X: sca.transform(imp.transform(X))

    pre = members()
    for m in pre:
        m.fit(T(Xa), ya)

    Xtt, ytt = xy(TSU, S["tsu_test"])
    Xfe, yfe = xy(FP, S["fp_test"])

    Xft, yft = xy(FP, S["fp_finetune"])
    Xfa, yfa = augment(Xft, yft)

    # D: fine-tune the Tsuboyama model on FireProt (reuse Tsuboyama imputer/scaler).
    ft = copy.deepcopy(pre)
    for m in ft:
        m.set_params(learning_rate_init=1e-3, max_iter=400)
        m.fit(T(Xfa), yfa)

    # B: FireProt-only baseline — a fresh model trained on FireProt alone, with its
    # own imputer/scaler (how good is FireProt without any Tsuboyama pretraining?).
    impB = SimpleImputer(strategy="median").fit(Xfa)
    scaB = StandardScaler().fit(impB.transform(Xfa))
    TB = lambda X: scaB.transform(impB.transform(X))
    bms = members()
    for m in bms:
        m.fit(TB(Xfa), yfa)

    for cond, ms, Tf in (("A_tsu_only", pre, T), ("B_fp_only", bms, TB),
                         ("D_finetuned", ft, T)):
        for tag, X, y in (("tsu_test", Xtt, ytt), ("fp_test", Xfe, yfe)):
            m = compute_metrics(y, predict(ms, Tf, X))
            rows.append({"thr": NN, "cond": cond, "test": tag, **m})
            print(f"thr={NN} {cond:11s} {tag:8s} r={m['pearson']:.3f} "
                  f"rho={m['spearman']:.3f} RMSE={m['rmse']:.3f} n={m['n']}", flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
print("\n=== FireProt-test Pearson (does finetuning help?) ===")
print(res[res.test == "fp_test"].pivot_table(index="thr", columns="cond",
      values="pearson").round(3).to_string())
print("\n=== Tsuboyama-test Pearson (forgetting?) ===")
print(res[res.test == "tsu_test"].pivot_table(index="thr", columns="cond",
      values="pearson").round(3).to_string())
print(f"\nwrote {OUT/'results.csv'}")
