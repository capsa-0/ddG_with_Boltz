"""Where does the S669 error concentrate, by mutation class?

Burial proxy comes from Boltz's own predicted distogram row at the mutated position
(`pdrow`, 64 bins over 2-22 A): expected number of residues within 10 A, |i-j|>2.

Sign convention: S669 file stores ddG with POSITIVE = stabilizing (run_benchmarks
sign-flips predictions). Here everything is converted to the standard
POSITIVE = DESTABILIZING convention, which is what the model was trained on.
"""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCR = ROOT / "data/processed/_analysis"
SCR.mkdir(parents=True, exist_ok=True)
Z, N_SEED = 128, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]

VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1,
           H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0,
           T=116.1, W=227.8, Y=193.6, V=140.0)
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)


def mat(df):
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def augment(X, y):
    return (np.vstack([X, np.concatenate([X[:, Z:], X[:, :Z]], axis=1)]),
            np.concatenate([y, -y]))


def members():
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=250, early_stopping=False,
                         random_state=s, warm_start=True) for s in range(N_SEED)]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-9 else np.nan


def rho(a, b):
    return r(pd.Series(a).rank(), pd.Series(b).rank())


# ---------------------------------------------------------------- burial from pdrow
def contact_numbers(slim_dir, cutoff=10.0):
    bounds = np.linspace(2.0, 22.0, 63)          # 63 boundaries -> 64 bins
    keep = np.concatenate([[True], bounds < cutoff])   # bin k is below cutoff
    out = {}
    for f in sorted(Path(slim_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        for i, k in enumerate(d["keys"]):
            pr = d[f"pdrow_{i}"]
            if pr.ndim == 3:
                pr = pr[-1]
            pr = pr.astype(np.float64)
            p = np.exp(pr - pr.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)          # softmax over 64 bins
            pos = int(d[f"pos_{i}"].ravel()[0])
            near = p[:, keep].sum(-1)              # P(d < cutoff) per residue j
            idx = np.arange(len(near))
            near = near[np.abs(idx - pos) > 2]     # drop sequence neighbours
            out[str(k)] = float(near.sum())
    return out


def main():
    tsu = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
    fp = pd.read_parquet(ROOT / "data/processed/fireprot_le500/features_ablation.parquet")
    bdf = pd.read_parquet(ROOT / "data/processed/s669/features_ablation.parquet")

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

    X = mat(bdf)
    df = bdf[["wt_id", "mutation", "ddg"]].copy()
    df["y"] = -df["ddg"].astype(float)                     # -> positive = destabilizing
    for name, ms, T in (("A", A, TA), ("B", B, TB), ("D", D, TA)):
        df[f"pred_{name}"] = np.mean([m.predict(T(X)) for m in ms], axis=0)
    df.to_csv(SCR / "s669_predictions.csv", index=False)

    df["wt_aa"] = df.mutation.str[0]
    df["mut_aa"] = df.mutation.str[-1]
    df["dVol"] = df.mut_aa.map(VOL) - df.wt_aa.map(VOL)
    df["dKD"] = df.mut_aa.map(KD) - df.wt_aa.map(KD)

    cn = contact_numbers(ROOT / "data/processed/s669/slim")
    key = bdf["wt_id"].astype(str) + "_" + bdf["mutation"].astype(str)
    df["burial"] = [cn.get(k, np.nan) for k in key]
    if df.burial.isna().mean() > 0.5:   # key format fallback
        df["burial"] = [cn.get(k, np.nan) for k in bdf.get("sample_key", key).astype(str)]
    print(f"burial resolved for {df.burial.notna().mean():.0%} of variants "
          f"(median {df.burial.median():.1f} contacts)\n")

    R = "D"
    df["err"] = df[f"pred_{R}"] - df.y                     # signed: >0 = over-predicts
    # remove the per-protein offset so class effects are not swamped by it
    df["err_c"] = df.err - df.groupby("wt_id")["err"].transform("mean")

    print(f"=== regime {R}: overall  r={r(df.y, df[f'pred_{R}']):.3f}  "
          f"rho={rho(df.y, df[f'pred_{R}']):.3f}  MAE={df.err.abs().mean():.2f}  n={len(df)}\n")

    def tbl(g, label, minn=8):
        rows = []
        for k, s in g:
            if len(s) < minn:
                continue
            rows.append({label: k, "n": len(s),
                         "bias": s.err.mean(), "MAE": s.err.abs().mean(),
                         "MAE_c": s.err_c.abs().mean(),
                         "rho": rho(s.y, s[f"pred_{R}"]),
                         "sd_true": s.y.std()})
        return pd.DataFrame(rows).sort_values("MAE", ascending=False)

    print("--- by WT residue (mutating AWAY from) ---")
    print(tbl(df.groupby("wt_aa"), "wt").round(2).to_string(index=False))
    print("\n--- by mutant residue (mutating TO) ---")
    print(tbl(df.groupby("mut_aa"), "mut").round(2).to_string(index=False))

    df["cls"] = np.where(df.wt_aa == "G", "from Gly",
                np.where(df.mut_aa == "G", "to Gly",
                np.where(df.wt_aa == "P", "from Pro",
                np.where(df.mut_aa == "P", "to Pro", "other"))))
    print("\n--- Gly / Pro classes ---")
    print(tbl(df.groupby("cls"), "class", minn=5).round(2).to_string(index=False))

    df["burial_q"] = pd.qcut(df.burial, 3, labels=["exposed", "mid", "buried"])
    print("\n--- by burial (Boltz distogram contact number, tertiles) ---")
    print(tbl(df.groupby("burial_q", observed=True), "burial").round(2).to_string(index=False))

    df["dir"] = np.where(df.y > 0.5, "destabilizing", np.where(df.y < -0.5, "stabilizing", "neutral"))
    print("\n--- by true effect direction ---")
    print(tbl(df.groupby("dir"), "direction", minn=5).round(2).to_string(index=False))

    df["volq"] = pd.qcut(df.dVol, 3, labels=["smaller", "similar", "larger"])
    print("\n--- by volume change ---")
    print(tbl(df.groupby("volq", observed=True), "dVol").round(2).to_string(index=False))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    rng = np.random.default_rng(0)
    C = "#0E6C68"

    ax = axes[0]
    order = ["from Gly", "to Gly", "from Pro", "to Pro", "other"]
    data = [df.loc[df.cls == k, "err_c"].to_numpy() for k in order]
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for p in bp["boxes"]:
        p.set(facecolor=C, alpha=0.25, edgecolor=C, lw=1.2)
    for i, k in enumerate(order):
        v = df.loc[df.cls == k, "err_c"].to_numpy()
        ax.scatter(1 + i + rng.uniform(-0.17, 0.17, len(v)), v, s=11, color=C,
                   alpha=0.5, edgecolors="none", zorder=3)
    ax.axhline(0, color="#444", lw=1.1, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{k}\nn={int((df.cls == k).sum())}" for k in order], fontsize=8.5)
    ax.set_ylabel("protein-centred error  (pred − true)  kcal/mol", fontsize=9)
    ax.set_title("Gly / Pro substitutions", fontsize=10.5, pad=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    ax = axes[1]
    order2 = ["exposed", "mid", "buried"]
    data = [df.loc[df.burial_q == k, "err_c"].dropna().to_numpy() for k in order2]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for p in bp["boxes"]:
        p.set(facecolor="#9A5B0C", alpha=0.25, edgecolor="#9A5B0C", lw=1.2)
    for i, k in enumerate(order2):
        v = df.loc[df.burial_q == k, "err_c"].dropna().to_numpy()
        ax.scatter(1 + i + rng.uniform(-0.16, 0.16, len(v)), v, s=11, color="#9A5B0C",
                   alpha=0.5, edgecolors="none", zorder=3)
    ax.axhline(0, color="#444", lw=1.1, ls="--")
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([f"{k}\nn={int((df.burial_q == k).sum())}" for k in order2], fontsize=8.5)
    ax.set_ylabel("protein-centred error  kcal/mol", fontsize=9)
    ax.set_title("Burial (Boltz distogram contact number)", fontsize=10.5, pad=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    ax = axes[2]
    order3 = ["stabilizing", "neutral", "destabilizing"]
    for i, k in enumerate(order3):
        s = df[df.dir == k]
        ax.scatter(s.y, s[f"pred_{R}"], s=18, alpha=0.6, edgecolors="none",
                   label=f"{k} (n={len(s)})")
    lim = [df.y.min() - 0.4, df.y.max() + 0.4]
    ax.plot(lim, lim, color="#444", lw=1.1, ls="--", zorder=0)
    sl, ic = np.polyfit(df.y, df[f"pred_{R}"], 1)
    ax.plot(lim, [sl * v + ic for v in lim], color="#B3261E", lw=1.6,
            label=f"fit slope {sl:.2f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true ΔΔG  kcal/mol  (+ = destabilizing)", fontsize=9)
    ax.set_ylabel("predicted ΔΔG  kcal/mol", fontsize=9)
    ax.set_title("Amplitude compression", fontsize=10.5, pad=10)
    ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    fig.suptitle(f"S669 error by mutation class — regime {R} (Tsuboyama→FireProt fine-tuned), "
                 "protein offset removed", fontsize=11.5, y=1.0)
    fig.tight_layout()
    fig.savefig(SCR / "mut_class_error.png", dpi=190, bbox_inches="tight", facecolor="white")
    df.to_csv(SCR / "s669_mut_classes.csv", index=False)
    print(f"\nwrote {SCR/'mut_class_error.png'}")


if __name__ == "__main__":
    main()
