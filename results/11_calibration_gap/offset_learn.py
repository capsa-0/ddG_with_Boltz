"""Part 2: (a) train the per-protein offset head on the real training corpora
(550 proteins, not 61 LOPO), (b) per-protein error boxplots."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCR = ROOT / "data/processed/_analysis"
SCR.mkdir(parents=True, exist_ok=True)
Z, N_SEED = 128, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]
WT = [f"wtz_{j}" for j in range(Z)]


def mat(df):
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def augment(X, y):
    return (np.vstack([X, np.concatenate([X[:, Z:], X[:, :Z]], axis=1)]),
            np.concatenate([y, -y]))


def members():
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=250, early_stopping=False,
                         random_state=s, warm_start=True) for s in range(N_SEED)]


def predict(ms, T, X):
    return np.mean([m.predict(T(X)) for m in ms], axis=0)


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-9 else np.nan


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


tsu = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
fp = pd.read_parquet(ROOT / "data/processed/fireprot_le500/features_ablation.parquet")
bdf = pd.read_parquet(ROOT / "data/processed/s669/features_ablation.parquet")
leak = pd.read_csv(ROOT / "results/09_external_benchmarks/homology/s669_leakage.csv").set_index("protein")

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
keep25 = np.array([not bool(leak.loc[p, "leaky_any_25"]) if p in leak.index else True
                   for p in prot])

REG = {"A_tsu_only": (A, TA), "B_fp_only": (B, TB), "D_finetuned": (D, TA)}
preds = {}
for name, (ms, T) in REG.items():
    p0 = predict(ms, T, X)
    preds[name] = -p0 if r(y, p0) < 0 else p0

# ---------------------------------------------------------------- (a) learn the offset
print("=== OFFSET HEAD TRAINED ON THE REAL CORPORA (550 proteins) ===")
print("target: per-protein mean residual (y - pred); features: protein-mean WT embedding\n")

for name in ("B_fp_only", "D_finetuned"):
    ms, T = REG[name]
    # build protein-level training set from Tsuboyama + FireProt (in-sample residuals)
    tr_parts = []
    for src in (tsu, fp):
        ps = predict(ms, T, mat(src))
        g = pd.DataFrame({"p": src["wt_id"].to_numpy(), "res": src["ddg"].to_numpy(float) - ps})
        need = g.groupby("p")["res"].mean()
        desc = src.groupby("wt_id")[WT].mean().loc[need.index]
        tr_parts.append((np.nan_to_num(desc.to_numpy(float)), need.to_numpy(float)))
    Xtr = np.vstack([a for a, _ in tr_parts])
    ytr = np.concatenate([b for _, b in tr_parts])

    g = pd.DataFrame({"p": prot, "res": y - preds[name]})
    need_te = g.groupby("p")["res"].mean()
    desc_te = bdf.groupby("wt_id")[WT].mean().loc[need_te.index]
    Xte, yte = np.nan_to_num(desc_te.to_numpy(float)), need_te.to_numpy(float)

    sc = StandardScaler().fit(Xtr)
    for tag, mdl in (("ridge", RidgeCV(alphas=np.logspace(-2, 4, 25))),
                     ("rf", RandomForestRegressor(400, min_samples_leaf=3, random_state=0))):
        mdl.fit(sc.transform(Xtr), ytr)
        ph = mdl.predict(sc.transform(Xte))
        off = dict(zip(need_te.index, ph))
        corr = preds[name] + np.array([off[q] for q in prot])
        print(f"  {name:12s} {tag:5s} | n_train_prot={len(ytr)}  offset r={r(yte, ph):+.3f}  "
              f"RMSE {rmse(yte, ph):.2f} (mean-baseline {rmse(yte, np.full_like(yte, ytr.mean())):.2f})")
        print(f"               -> pooled r full {r(y, preds[name]):.3f} -> {r(y, corr):.3f} | "
              f"common25 {r(y[keep25], preds[name][keep25]):.3f} -> {r(y[keep25], corr[keep25]):.3f}")

# ---------------------------------------------------------------- (b) boxplots
rows = []
for name in REG:
    e = y - preds[name]
    for q in np.unique(prot):
        m = prot == q
        rows.append({"regime": name, "protein": q, "n": int(m.sum()),
                     "mean_err": float(e[m].mean()), "mae": float(np.abs(e[m]).mean()),
                     "true_mean_ddg": float(y[m].mean()),
                     "clean": bool(keep25[m][0])})
pp = pd.DataFrame(rows)
pp.to_csv(SCR / "per_protein_error.csv", index=False)

LBL = {"A_tsu_only": "A\nTsuboyama", "B_fp_only": "B\nFireProt", "D_finetuned": "D\nfine-tuned"}
order = ["A_tsu_only", "B_fp_only", "D_finetuned"]
C = {"A_tsu_only": "#0E6C68", "B_fp_only": "#9A5B0C", "D_finetuned": "#4B5D9A"}

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
rng = np.random.default_rng(0)

for ax, col, ttl, yl in (
        (axes[0], "mean_err", "Per-protein mean signed error\n(the offset term)", "mean(y − ŷ)  kcal/mol"),
        (axes[1], "mae", "Per-protein mean absolute error", "MAE  kcal/mol")):
    data = [pp.loc[pp.regime == k, col].to_numpy() for k in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.6))
    for patch, k in zip(bp["boxes"], order):
        patch.set(facecolor=C[k], alpha=0.28, edgecolor=C[k], lw=1.3)
    for i, k in enumerate(order):
        v = pp.loc[pp.regime == k, col].to_numpy()
        ax.scatter(1 + i + rng.uniform(-0.16, 0.16, len(v)), v, s=13, color=C[k],
                   alpha=0.65, edgecolors="none", zorder=3)
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([LBL[k] for k in order], fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.set_title(ttl, fontsize=10.5, pad=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    if col == "mean_err":
        ax.axhline(0, color="#444", lw=1.1, ls="--", zorder=1)
        for i, k in enumerate(order):
            v = pp.loc[pp.regime == k, "mean_err"].to_numpy()
            ax.text(1 + i, ax.get_ylim()[1] * 0.93, f"sd {v.std():.2f}",
                    ha="center", fontsize=8, color=C[k])

ax = axes[2]
d = pp[pp.regime == "D_finetuned"]
ax.scatter(d.true_mean_ddg, d.mean_err, s=26, color=C["D_finetuned"], alpha=0.75,
           edgecolors="none")
lo, hi = d.true_mean_ddg.min(), d.true_mean_ddg.max()
sl, ic = np.polyfit(d.true_mean_ddg, d.mean_err, 1)
xs = np.linspace(lo, hi, 50)
ax.plot(xs, sl * xs + ic, color="#B3261E", lw=1.6,
        label=f"slope {sl:.2f}, r={r(d.true_mean_ddg.to_numpy(), d.mean_err.to_numpy()):.2f}")
ax.axhline(0, color="#444", lw=1.1, ls="--")
ax.set_xlabel("protein's true mean ΔΔG  kcal/mol", fontsize=9)
ax.set_ylabel("mean(y − ŷ)  kcal/mol", fontsize=9)
ax.set_title("Offset is systematic, not noise\n(regime D)", fontsize=10.5, pad=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

fig.suptitle("S669 per-protein error — where the pooled correlation is lost  (62 proteins, 541 variants)",
             fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig(SCR / "per_protein_error.png", dpi=190, bbox_inches="tight",
            facecolor="white")
print(f"\nwrote {SCR/'per_protein_error.png'}")
print("\n=== per-protein mean signed error, summary ===")
print(pp.groupby("regime")["mean_err"].agg(["mean", "std", "min", "max"]).round(3).to_string())
print("\n=== per-protein MAE, summary ===")
print(pp.groupby("regime")["mae"].agg(["median", "mean", "max"]).round(3).to_string())
print("\nworst-offset proteins (regime D):")
print(d.reindex(d.mean_err.abs().sort_values(ascending=False).index)
      [["protein", "n", "true_mean_ddg", "mean_err", "mae"]].head(8).round(2).to_string(index=False))
