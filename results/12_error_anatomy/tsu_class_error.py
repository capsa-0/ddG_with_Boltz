"""Mutation-class error breakdown on HELD-OUT Tsuboyama.

Two held-out views, neither of which the scoring model was trained on:
  OOF : out-of-fold predictions from 5-fold GroupKFold on wt_id (the results/06 protocol)
  B   : the FireProt-only model, which never saw any Tsuboyama at all

Burial proxy from Boltz's own distogram row at the mutated position (64 bins, 2-22 A):
expected number of residues within 10 A, |i-j| > 2.
Convention: positive ddG = destabilizing (as stored in Tsuboyama).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCR = ROOT / "data/processed/_analysis"
SCR.mkdir(parents=True, exist_ok=True)
Z, N_SEED, N_FOLD = 128, 5, 5
FEAT = [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)]

VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1,
           H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0,
           T=116.1, W=227.8, Y=193.6, V=140.0)


def mat(df):
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def augment(X, y):
    return (np.vstack([X, np.concatenate([X[:, Z:], X[:, :Z]], axis=1)]),
            np.concatenate([y, -y]))


def members(n=N_SEED):
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=250, early_stopping=False,
                         random_state=s, warm_start=True) for s in range(n)]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-9 else np.nan


def rho(a, b):
    return r(pd.Series(a).rank(), pd.Series(b).rank())


def contact_numbers(slim_dir, cutoff=10.0):
    bounds = np.linspace(2.0, 22.0, 63)
    keep = np.concatenate([[True], bounds < cutoff])
    out = {}
    for f in sorted(Path(slim_dir).glob("*.npz")):
        d = np.load(f, allow_pickle=True)
        for i, k in enumerate(d["keys"]):
            pr = d[f"pdrow_{i}"]
            if pr.ndim == 3:
                pr = pr[-1]
            pr = pr.astype(np.float64)
            p = np.exp(pr - pr.max(-1, keepdims=True))
            p /= p.sum(-1, keepdims=True)
            pos = int(d[f"pos_{i}"].ravel()[0])
            near = p[:, keep].sum(-1)
            idx = np.arange(len(near))
            out[str(k)] = float(near[np.abs(idx - pos) > 2].sum())
    return out


def main():
    tsu = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
    fp = pd.read_parquet(ROOT / "data/processed/fireprot_le500/features_ablation.parquet")
    X, y = mat(tsu), tsu["ddg"].to_numpy(float)
    groups = tsu["wt_id"].to_numpy()
    print(f"Tsuboyama {len(tsu)} muts / {tsu.wt_id.nunique()} proteins", flush=True)

    # ---- out-of-fold, GroupKFold on protein ----
    oof = np.full(len(y), np.nan)
    gkf = GroupKFold(n_splits=N_FOLD)
    for fi, (tr, te) in enumerate(gkf.split(X, y, groups), 1):
        Xa, ya = augment(X[tr], y[tr])
        imp = SimpleImputer(strategy="median").fit(Xa)
        sca = StandardScaler().fit(imp.transform(Xa))
        T = lambda Q: sca.transform(imp.transform(Q))
        ms = members()
        for m in ms:
            m.fit(T(Xa), ya)
        oof[te] = np.mean([m.predict(T(X[te])) for m in ms], axis=0)
        print(f"  fold {fi}/{N_FOLD}: n_test={len(te)} r={r(y[te], oof[te]):.3f}", flush=True)

    # ---- FireProt-only model (never saw Tsuboyama) ----
    Xf, yf = mat(fp), fp["ddg"].to_numpy(float)
    Xfa, yfa = augment(Xf, yf)
    impB = SimpleImputer(strategy="median").fit(Xfa)
    scaB = StandardScaler().fit(impB.transform(Xfa))
    TB = lambda Q: scaB.transform(impB.transform(Q))
    B = members()
    for m in B:
        m.fit(TB(Xfa), yfa)
    predB = np.mean([m.predict(TB(X)) for m in B], axis=0)
    if r(y, predB) < 0:
        predB = -predB

    df = tsu[["wt_id", "mutation", "ddg"]].copy()
    df["y"] = y
    df["pred_OOF"] = oof
    df["pred_B"] = predB
    df["wt_aa"] = df.mutation.str[0]
    df["mut_aa"] = df.mutation.str[-1]
    df["dVol"] = df.mut_aa.map(VOL) - df.wt_aa.map(VOL)
    df.to_csv(SCR / "tsu_mut_classes.csv", index=False)   # save before analysis/plotting

    cn = contact_numbers(ROOT / "data/processed/tsuboyama_bench_fast/slim")
    for cand in (df.wt_id.astype(str) + "_" + df.mutation.astype(str),
                 tsu.get("sample_key", pd.Series(index=tsu.index, dtype=object)).astype(str)):
        b = [cn.get(k, np.nan) for k in cand]
        if np.mean([np.isfinite(v) for v in b]) > 0.5:
            df["burial"] = b
            break
    else:
        df["burial"] = np.nan
    print(f"\nburial resolved for {df.burial.notna().mean():.0%} "
          f"(median {df.burial.median():.1f} contacts)", flush=True)

    print(f"\n=== OOF overall: r={r(df.y, df.pred_OOF):.3f} rho={rho(df.y, df.pred_OOF):.3f} "
          f"MAE={np.abs(df.pred_OOF - df.y).mean():.2f}  n={len(df)}")
    print(f"=== B   overall: r={r(df.y, df.pred_B):.3f} rho={rho(df.y, df.pred_B):.3f} "
          f"MAE={np.abs(df.pred_B - df.y).mean():.2f}\n")

    R = "OOF"
    df["err"] = df[f"pred_{R}"] - df.y
    df["err_c"] = df.err - df.groupby("wt_id")["err"].transform("mean")

    def tbl(g, label, minn=40):
        rows = []
        for k, s in g:
            if len(s) < minn:
                continue
            rows.append({label: k, "n": len(s), "bias": s.err.mean(),
                         "MAE": s.err.abs().mean(), "MAE_c": s.err_c.abs().mean(),
                         "rho": rho(s.y, s[f"pred_{R}"]), "sd_true": s.y.std()})
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
    print(tbl(df.groupby("cls"), "class", minn=20).round(2).to_string(index=False))

    df["burial_q"] = pd.qcut(df.burial, 3, labels=["exposed", "mid", "buried"])
    print("\n--- by burial (tertiles) ---")
    print(tbl(df.groupby("burial_q", observed=True), "burial").round(2).to_string(index=False))

    df["dir"] = np.where(df.y > 0.5, "destabilizing",
                         np.where(df.y < -0.5, "stabilizing", "neutral"))
    print("\n--- by true effect direction ---")
    print(tbl(df.groupby("dir"), "direction", minn=20).round(2).to_string(index=False))

    df["volq"] = pd.qcut(df.dVol, 3, labels=["smaller", "similar", "larger"])
    print("\n--- by volume change ---")
    print(tbl(df.groupby("volq", observed=True), "dVol").round(2).to_string(index=False))

    print("\n--- burial x Gly (the interaction of interest) ---")
    sub = df[df.burial.notna()].copy()
    sub["gly"] = np.where(sub.wt_aa == "G", "from Gly", "non-Gly")
    print(tbl(sub.groupby(["burial_q", "gly"], observed=True), "cell", minn=20)
          .round(2).to_string(index=False))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
    C1, C2 = "#0E6C68", "#9A5B0C"

    ax = axes[0]
    order = ["from Gly", "to Gly", "from Pro", "to Pro", "other"]
    bp = ax.boxplot([df.loc[df.cls == k, "err_c"].to_numpy() for k in order],
                    patch_artist=True, widths=0.6, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for p in bp["boxes"]:
        p.set(facecolor=C1, alpha=0.25, edgecolor=C1, lw=1.2)
    ax.axhline(0, color="#444", lw=1.1, ls="--")
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels([f"{k}\nn={int((df.cls == k).sum())}" for k in order], fontsize=8.5)
    ax.set_ylabel("protein-centred error (pred − true)  kcal/mol", fontsize=9)
    ax.set_title("Gly / Pro substitutions", fontsize=10.5, pad=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    ax = axes[1]
    order2 = ["exposed", "mid", "buried"]
    bp = ax.boxplot([df.loc[df.burial_q == k, "err_c"].dropna().to_numpy() for k in order2],
                    patch_artist=True, widths=0.55, showfliers=False,
                    medianprops=dict(color="black", lw=1.5))
    for p in bp["boxes"]:
        p.set(facecolor=C2, alpha=0.25, edgecolor=C2, lw=1.2)
    ax.axhline(0, color="#444", lw=1.1, ls="--")
    ax.set_xticks(range(1, 4))
    ax.set_xticklabels([f"{k}\nn={int((df.burial_q == k).sum())}" for k in order2], fontsize=8.5)
    ax.set_ylabel("protein-centred error  kcal/mol", fontsize=9)
    ax.set_title("Burial (Boltz distogram contact number)", fontsize=10.5, pad=10)
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    ax = axes[2]
    for k in ["stabilizing", "neutral", "destabilizing"]:
        s = df[df.dir == k]
        ax.scatter(s.y, s.pred_OOF, s=5, alpha=0.25, edgecolors="none", label=f"{k} (n={len(s)})")
    lim = [np.nanpercentile(df.y, 0.2) - 0.5, np.nanpercentile(df.y, 99.8) + 0.5]
    ax.plot(lim, lim, color="#444", lw=1.1, ls="--", zorder=0)
    sl, ic = np.polyfit(df.y, df.pred_OOF, 1)
    ax.plot(lim, [sl * v + ic for v in lim], color="#B3261E", lw=1.6, label=f"fit slope {sl:.2f}")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("true ΔΔG kcal/mol (+ = destabilizing)", fontsize=9)
    ax.set_ylabel("predicted ΔΔG kcal/mol", fontsize=9)
    ax.set_title("Amplitude compression", fontsize=10.5, pad=10)
    lg = ax.legend(fontsize=7.5, frameon=False, loc="upper left")
    for h in lg.legend_handles:
        if hasattr(h, "set_sizes"):
            h.set_sizes([26])
        h.set_alpha(0.9)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)

    fig.suptitle("Held-out Tsuboyama (5-fold protein GroupKFold, out-of-fold) — "
                 f"error by mutation class, protein offset removed  (n={len(df)})",
                 fontsize=11.5, y=1.0)
    fig.tight_layout()
    fig.savefig(SCR / "tsu_mut_class_error.png", dpi=190, bbox_inches="tight", facecolor="white")
    df.to_csv(SCR / "tsu_mut_classes.csv", index=False)
    print(f"\nwrote {SCR/'tsu_mut_class_error.png'}")


if __name__ == "__main__":
    main()
