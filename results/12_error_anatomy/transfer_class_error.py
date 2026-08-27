"""Error anatomy by substitution class, ON TRANSFER, with the transfer-best readout.

Why this exists alongside `mut_class_error.py`: that script's S669 breakdown used the
**concat + regime D** predictor, which results/14 later measured as the *worst* readout
on transfer (S669 r 0.476 vs 0.557 for the pair-track diagonal), and S669 alone has no
power for class cells (`from-Pro` n=5, `->Pro` n=6). FireProt -- the one blind corpus
with power (3,102 variants / 130 proteins after homology filtering) -- had never been
broken down by class at all.

This script re-does the breakdown on **both** blind corpora with the adopted transfer
readout (`ddg.evaluation.labels.TRANSFER_BLOCKS` = the diagonal alone). It re-uses the
per-variant prediction dumps results/14 already wrote, so it re-trains nothing and needs
no GPU:

    data/processed/_analysis/exp14_s669_results_s669_locality.csv    (col `diag`)
    data/processed/_analysis/exp14_s669_results_onehot_s669.csv      (col `onehot`)
    data/processed/_analysis/exp14_fpfilt_results_locality_paired.csv
    data/processed/_analysis/exp14_fpfilt_results_onehot_fp.csv

The `onehot` column is the substitution-identity control (40 one-hot dims) from
results/14. Carrying it lets us ask the question the original breakdown could not: not
just "where is the error largest" but **"where does the embedding beat a plain amino-acid
lookup"** -- `skill = 1 - MAE_diag / MAE_onehot`.

Conventions kept from `mut_class_error.py`:
  - ddG POSITIVE = destabilizing (results/14 uses ddg < -0.5 for the stabilizing tail).
  - Error is reported raw **and protein-centred** (per-protein mean error removed), so
    class effects are not confounded with the calibration gap of results/11.
  - Class MAE is read against that class's own sd of true ddG, else classes holding
    bigger effects look "harder" by arithmetic alone.
  - CIs are a **cluster bootstrap over proteins**, the standard of evidence in this
    series.

DATA GOTCHA: S669 contains **17 repeated (protein, mutation) keys with different measured
ddG** -- genuine repeat measurements in the benchmark. Merging the dumps on that key
cross-products them and inflates 541 -> 575 rows. The dumps are row-aligned, so we merge
**by position** and assert the keys agree.

    python results/12_error_anatomy/transfer_class_error.py

Writes `transfer_class_tables.csv`, `transfer_class_bootstrap.csv` and
`figures/03_transfer_class_error.png` into this folder.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
HERE = ROOT / "results/12_error_anatomy"
DUMPS = ROOT / "data/processed/_analysis"
N_BOOT = 600
RNG = np.random.default_rng(0)

# CVD-validated palette adopted in results/14 (the repo's older gray-green/teal pair
# fails adjacent-pair separation at deltaE 2.3 under deuteranopia).
BLUE, ORANGE, GREEN, PURPLE = "#1F6FB4", "#D95F02", "#1B9E77", "#7570B3"

VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1,
           H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0,
           T=116.1, W=227.8, Y=193.6, V=140.0)

CORPORA = {
    "s669": dict(label="S669", diag="exp14_s669_results_s669_locality.csv",
                 onehot="exp14_s669_results_onehot_s669.csv", min_res=12, min_pair=6),
    "fpfilt": dict(label="FireProt <=500 (filt)", diag="exp14_fpfilt_results_locality_paired.csv",
                   onehot="exp14_fpfilt_results_onehot_fp.csv", min_res=40, min_pair=10),
}


# --------------------------------------------------------------------- loading
def load(tag: str) -> pd.DataFrame:
    """Per-variant predictions for one blind corpus, with derived error columns."""
    spec = CORPORA[tag]
    d = pd.read_csv(DUMPS / spec["diag"])[["wt_id", "mutation", "ddg", "diag"]]
    oh = pd.read_csv(DUMPS / spec["onehot"])
    # Positional merge -- see the DATA GOTCHA in the module docstring.
    assert len(d) == len(oh), f"{tag}: dump lengths differ"
    assert (d.wt_id.values == oh.wt_id.values).all(), f"{tag}: wt_id order differs"
    assert (d.mutation.values == oh.mutation.values).all(), f"{tag}: mutation order differs"
    d = d.copy()
    d["onehot"] = oh.onehot.values
    d["wt_aa"] = d.mutation.str[0]
    d["mut_aa"] = d.mutation.str[-1]
    d["dvol"] = d.mut_aa.map(VOL) - d.wt_aa.map(VOL)
    d["err"] = d["diag"] - d["ddg"]
    d["err_oh"] = d["onehot"] - d["ddg"]
    # Protein-centred error: strips the per-protein calibration offset (results/11), so a
    # class effect cannot be "this class happens to sit in badly-calibrated proteins".
    d["err_c"] = d["err"] - d.groupby("wt_id")["err"].transform("mean")
    return d


# --------------------------------------------------------------------- metrics
def stats(g: pd.DataFrame) -> pd.Series:
    y, e, ec, eo = g.ddg.values, g.err.values, g.err_c.values, g.err_oh.values
    sd = y.std(ddof=1) if len(y) > 1 else np.nan
    mae, mae_oh = np.abs(e).mean(), np.abs(eo).mean()
    return pd.Series({
        "n": len(g),
        "MAE": mae,
        "MAE_centred": np.abs(ec).mean(),
        "bias": e.mean(),
        "sd_true": sd,
        "MAE_sd": mae / sd if sd and sd > 0 else np.nan,
        "rho": spearmanr(g.diag, y).correlation if len(y) >= 25 else np.nan,
        # skill > 0: the embedding beats a plain amino-acid lookup on this class.
        # skill ~ 0: the prediction has collapsed to a substitution-matrix average.
        "skill_vs_onehot": 1 - mae / mae_oh if mae_oh > 0 else np.nan,
    })


def table(d: pd.DataFrame, by: str, min_n: int) -> pd.DataFrame:
    t = d.groupby(by).apply(stats, include_groups=False)
    return t[t.n >= min_n].sort_values("MAE_sd", ascending=False)


def cluster_boot(d: pd.DataFrame, mask: pd.Series, stat, n_boot: int = N_BOOT):
    """Percentile CI for `stat(in_class, out_of_class)`, resampling whole proteins."""
    prots = d.wt_id.unique()
    idx = {p: np.flatnonzero(d.wt_id.values == p) for p in prots}
    m = mask.values
    out = []
    for _ in range(n_boot):
        rows = np.concatenate([idx[p] for p in RNG.choice(prots, len(prots), replace=True)])
        sub, sm = d.iloc[rows], m[rows]
        if sm.sum() < 10 or (~sm).sum() < 10:
            continue
        out.append(stat(sub[sm], sub[~sm]))
    if not out:
        return np.nan, np.nan
    return tuple(np.percentile(out, [2.5, 97.5]))


def d_mae_sd(a: pd.DataFrame, b: pd.DataFrame) -> float:
    """Class MAE/sd minus the rest's MAE/sd -- positive = the class is harder."""
    return (np.abs(a.err).mean() / a.ddg.std(ddof=1)
            - np.abs(b.err).mean() / b.ddg.std(ddof=1))


def classes_of(d: pd.DataFrame) -> dict:
    """The mutation classes the literature and results/12 flag as weak spots."""
    return {
        "->Pro": d.mut_aa == "P",
        "from Pro": d.wt_aa == "P",
        "from Gly": d.wt_aa == "G",
        "->Gly": d.mut_aa == "G",
        "from aromatic (FWY)": d.wt_aa.isin(list("FWY")),
        "X->Ala": d.mut_aa == "A",
        "near-isosteric (|dVol|<30)": d.dvol.abs() < 30,
    }


# ------------------------------------------------------------------------ main
def main() -> None:
    data = {tag: load(tag) for tag in CORPORA}
    rows, boots = [], []

    for tag, d in data.items():
        spec = CORPORA[tag]
        ov = stats(d)
        print(f"\n{'=' * 78}\n{spec['label']}  n={len(d)}  proteins={d.wt_id.nunique()}  "
              f"r={pearsonr(d.diag, d.ddg)[0]:.3f}  MAE={ov.MAE:.3f}  "
              f"MAE/sd={ov.MAE_sd:.3f}  skill={ov.skill_vs_onehot:+.3f}\n{'=' * 78}")

        # Overall + the isosteric split, so the report reads every number from a table.
        rows.append(ov.to_frame().T.assign(**{"class": "all", "corpus": spec["label"],
                                              "grouping": "overall"}))
        iso = d.dvol.abs() < 30
        for lab, sub in [("near-isosteric", d[iso]), ("rest", d[~iso])]:
            rows.append(stats(sub).to_frame().T.assign(**{"class": lab, "corpus": spec["label"],
                                                          "grouping": "isosteric"}))

        d = d.assign(pair=d.wt_aa + "->" + d.mut_aa)
        for by, min_n, title in [("wt_aa", spec["min_res"], "FROM residue"),
                                 ("mut_aa", spec["min_res"], "TO residue"),
                                 ("pair", spec["min_pair"], "substitution pair")]:
            t = table(d, by, min_n)
            print(f"\n--- {title} (n>={min_n}) ---")
            print(t.round(3).to_string())
            rows.append(t.reset_index().rename(columns={by: "class"})
                        .assign(corpus=spec["label"], grouping=by))

        # Class contrasts with a cluster bootstrap over proteins.
        print(f"\n--- class vs the rest, d(MAE/sd), cluster bootstrap over proteins ---")
        for name, m in classes_of(d).items():
            lo, hi = cluster_boot(d, m, d_mae_sd)
            delta = d_mae_sd(d[m], d[~m])
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {name:28s} n={int(m.sum()):5d}  d(MAE/sd)={delta:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}] {sig}")
            boots.append(dict(corpus=spec["label"], klass=name, n=int(m.sum()),
                              delta_mae_sd=delta, lo=lo, hi=hi, significant=sig.strip() == "*"))

    # ------------------------------------------------------------- replication
    print(f"\n{'=' * 78}\nCROSS-CORPUS REPLICATION (Spearman between the two blind corpora)\n{'=' * 78}")
    rep = {}
    for by in ("wt_aa", "mut_aa"):
        a = table(data["s669"], by, CORPORA["s669"]["min_res"])
        b = table(data["fpfilt"], by, CORPORA["fpfilt"]["min_res"])
        j = a.join(b, lsuffix="_s", rsuffix="_f", how="inner")
        for metric in ("MAE_sd", "rho", "skill_vs_onehot"):
            v = j[[f"{metric}_s", f"{metric}_f"]].dropna()
            if len(v) < 6:
                continue
            r = spearmanr(v.iloc[:, 0], v.iloc[:, 1])
            rep[(by, metric)] = (r.correlation, r.pvalue, len(v))
            print(f"  {by:7s} {metric:16s} k={len(v):2d}  "
                  f"Spearman = {r.correlation:+.2f} (p={r.pvalue:.3f})")

    # ----------------------------------------------------------- write + plot
    pd.concat(rows, ignore_index=True).to_csv(HERE / "transfer_class_tables.csv", index=False)
    pd.DataFrame(boots).to_csv(HERE / "transfer_class_bootstrap.csv", index=False)
    pd.DataFrame([dict(grouping=k[0], metric=k[1], spearman=v[0], p=v[1], k=v[2])
                  for k, v in rep.items()]).to_csv(HERE / "transfer_replication.csv", index=False)
    make_figure(data, pd.DataFrame(boots), rep)
    print(f"\nWrote transfer_class_tables.csv, transfer_class_bootstrap.csv, "
          f"transfer_replication.csv, figures/03_transfer_class_error.png")


def make_figure(data: dict, boots: pd.DataFrame, rep: dict) -> None:
    fp, s6 = data["fpfilt"], data["s669"]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6))

    # (A) forest plot of the class contrasts on the powered corpus
    ax = axes[0]
    b = boots[boots.corpus == CORPORA["fpfilt"]["label"]].iloc[::-1].reset_index(drop=True)
    y = np.arange(len(b))
    colors = [ORANGE if s else "#9aa0a6" for s in b.significant]
    ax.hlines(y, b.lo, b.hi, color=colors, lw=3, alpha=.85)
    ax.scatter(b.delta_mae_sd, y, color=colors, s=52, zorder=3)
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{k}  (n={n})" for k, n in zip(b.klass, b.n)], fontsize=9)
    ax.set_xlabel("Δ(MAE ÷ sd) vs all other mutations")
    ax.set_title("A — which classes are genuinely harder\n"
                 "FireProt ≤500 filtered (130 proteins), cluster bootstrap",
                 fontsize=10, loc="left")
    ax.set_xlim(min(-0.16, b.lo.min() - 0.04), b.hi.max() + 0.05)
    ax.text(0.98, 0.05, "orange = CI excludes 0", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=8, color=ORANGE)

    # (B) skill over the amino-acid lookup, per source residue, both corpora
    ax = axes[1]
    a = table(s6, "wt_aa", CORPORA["s669"]["min_res"])
    c = table(fp, "wt_aa", CORPORA["fpfilt"]["min_res"])
    j = a.join(c, lsuffix="_s", rsuffix="_f", how="inner").sort_values("skill_vs_onehot_f")
    yy = np.arange(len(j))
    ax.hlines(yy, j.skill_vs_onehot_s, j.skill_vs_onehot_f, color="#c9ccd1", lw=1.6, zorder=1)
    ax.scatter(j.skill_vs_onehot_s, yy, color=PURPLE, s=42, label="S669", zorder=3)
    ax.scatter(j.skill_vs_onehot_f, yy, color=BLUE, s=42, label="FireProt", zorder=3)
    ax.axvline(0, color="#444", lw=1)
    gl = fp.pipe(stats).skill_vs_onehot
    ax.axvline(gl, color=BLUE, ls=":", lw=1.2)
    ax.set_yticks(yy)
    ax.set_yticklabels(j.index, fontsize=9)
    ax.set_xlabel("skill = 1 − MAE(diagonal) ÷ MAE(one-hot substitution)")
    r, p, k = rep.get(("wt_aa", "skill_vs_onehot"), (np.nan,) * 3)
    ax.set_title("B — where the embedding beats an amino-acid lookup\n"
                 f"by source residue; cross-corpus Spearman {r:+.2f} (p={p:.2f}, k={k})",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.text(gl, len(j) - 0.4, " corpus mean", color=BLUE, fontsize=8, va="top")

    # (C) the same, per substitution pair -- the packing/chemistry split
    ax = axes[2]
    pairs = ["Y->F", "W->A", "W->F", "K->R", "T->S", "P->A", "A->G",
             "G->A", "V->A", "I->A", "L->A"]
    fpp = fp.assign(pair=fp.wt_aa + "->" + fp.mut_aa)
    s6p = s6.assign(pair=s6.wt_aa + "->" + s6.mut_aa)
    sk_f, sk_s, ns, keep = [], [], [], []
    for p_ in pairs:
        g = fpp[fpp.pair == p_]
        if len(g) < 10:
            continue
        keep.append(p_)
        ns.append(len(g))
        sk_f.append(stats(g).skill_vs_onehot)
        h = s6p[s6p.pair == p_]
        sk_s.append(stats(h).skill_vs_onehot if len(h) >= 8 else np.nan)
    yy = np.arange(len(keep))
    ax.barh(yy, sk_f, color=[GREEN if v > 0.12 else ORANGE for v in sk_f], alpha=.85)
    ok = ~np.isnan(sk_s)
    ax.scatter(np.array(sk_s)[ok], yy[ok], color=PURPLE, s=40, zorder=3,
               label="S669 (n≥8)")
    ax.axvline(0, color="#444", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{p_}  (n={n})" for p_, n in zip(keep, ns)], fontsize=9)
    ax.set_xlabel("skill = 1 − MAE(diagonal) ÷ MAE(one-hot substitution)")
    ax.set_title("C — chemistry-preserving vs core-packing substitutions\n"
                 "bars: FireProt; dots: S669", fontsize=10, loc="left")
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.set_xlim(min(-0.06, np.nanmin(sk_s) - 0.03), max(sk_f + [v for v in sk_s if v == v]) + 0.06)

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color="#e6e8ea", lw=.8)
        ax.set_axisbelow(True)

    fig.suptitle("Error anatomy on transfer, with the transfer-best readout "
                 "(pair-track diagonal, results/14)", fontsize=12, y=1.0)
    fig.tight_layout()
    out = HERE / "figures/03_transfer_class_error.png"
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
