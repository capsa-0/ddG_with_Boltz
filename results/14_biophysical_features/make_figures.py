"""Figures for 14_biophysical_features.

    python results/14_biophysical_features/make_figures.py

01 — what generalizes: the ladder of feature constructions on two blind corpora, the
     claims that replicated on both, and the in-distribution/transfer reversal.
02 — the three biology-motivated additions, all of which failed.

All transfer numbers use the NO-AUGMENTATION protocol so the two corpora are directly
comparable; FireProt is homology-filtered (8 proteins sharing a 30 %-identity cluster
with Tsuboyama removed -> 130 proteins). Colour identifies the EVALUATION SET
throughout both figures; every panel carries a legend or direct labels.

Palette validated with the dataviz CVD checker (worst adjacent pair dE 11.6 deutan /
19.9 normal vision; all six checks pass).
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "figures"
OUT.mkdir(exist_ok=True)

BLUE, ORANGE, GREEN, PURPLE = "#1F6FB4", "#D95F02", "#1B9E77", "#7570B3"
INK, MUTED, GRID, BAD = "#1a1a1a", "#555555", "#c9d1d9", "#b03a48"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#8b949e", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

res = pd.read_csv(HERE / "results_all.csv")


def get(cfg, st, col="r", aug=False):
    row = res[(res.config == cfg) & (res.set == st) & (res.augment == aug)]
    return float(row[col].iloc[0]) if len(row) else np.nan


def boot(stem, ref, cfg, metric):
    """(mean, lo, hi, significant) for one paired bootstrap difference."""
    f = HERE / f"bootstrap_{stem}_ref-{ref}.csv"
    if not f.exists():
        return np.nan, np.nan, np.nan, False
    d = pd.read_csv(f)
    d = d[(d.kind == "paired_diff") & (d.config == cfg) & (d.metric == metric)]
    if not len(d):
        return np.nan, np.nan, np.nan, False
    r = d.iloc[0]
    return float(r["mean"]), float(r.lo), float(r.hi), bool(r.significant)


FP_LOC = "exp14_fpfilt_results_locality_paired"
S6_LOC = "exp14_s669_results_s669_locality"
FP_FAR = "exp14_fpfilt_results_farctrl"
S6_OH = "exp14_s669_results_onehot_s669"
FP_OH = "exp14_fpfilt_results_onehot_fp"   # homology-filtered, like FP_LOC/FP_FAR
FP_CONS = "exp14_fpfilt_results_fp_cons"   # homology-filtered, like the rest

CORPUS = [("s669", "S669  (62 prot., leakage-free)", ORANGE),
          ("fireprot_le500", "FireProt ≤500  (130 prot., filtered)", GREEN)]

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.9))

ax = axes[0]
LADDER = [("onehot", "substitution identity\n(one-hot, 40d)"),
          ("far", "far-shell pooling\n(256d)"),
          ("base", "uniform pooling, levels\n(concat, 256d)"),
          ("cw", "contact-weighted levels\n(256d)"),
          ("dz", "diagonal + uniform diff\n(Δz, 256d)"),
          ("diag", "diagonal only\n(128d)"),
          ("dz_cw", "diagonal + contact diff\n(256d)")]
y = np.arange(len(LADDER))[::-1]
for (st, lbl, col), off in zip(CORPUS, (0.19, -0.19)):
    vals = [get(c, st) for c, _ in LADDER]
    ax.barh(y + off, vals, height=0.36, color=col, label=lbl, zorder=3)
    for v, yy in zip(vals, y + off):
        if np.isfinite(v):
            ax.annotate(f"{v:.3f}", (v, yy), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=7.6, color=INK)
ax.set_yticks(y)
ax.set_yticklabels([l for _, l in LADDER], fontsize=8)
ax.set_xlabel("Pearson r, blind transfer (no augmentation)", fontsize=8.8)
ax.set_xlim(0, 0.79)
ax.set_title("A · Local terms transfer; pooled levels do not", fontsize=10.5, pad=10)
hA, lA = ax.get_legend_handles_labels()
ax.grid(axis="x", alpha=0.3, lw=0.6, color=GRID)
ax.set_axisbelow(True)

ax = axes[1]
CLAIMS = [
    ("near shell − far shell", [(S6_LOC, "far", "cw"), (FP_FAR, "far", "cw")]),
    ("diagonal − substitution identity", [(S6_OH, "onehot", "diag"), (FP_OH, "onehot", "diag")]),
    ("diagonal − Δz   (256d → 128d)", [(S6_LOC, "dz", "diag"), (FP_LOC, "dz", "diag")]),
    ("contact weighting − uniform\n(inside the difference form)",
     [(S6_LOC, "dz", "dz_cw"), (FP_LOC, "dz", "dz_cw")]),
]
y = np.arange(len(CLAIMS))[::-1]
for ci, (_, specs) in enumerate(CLAIMS):
    for (stem, ref, cfg), (st, lbl, col), off in zip(specs, CORPUS, (0.18, -0.18)):
        m, lo, hi, sig = boot(stem, ref, cfg, "r")
        if not np.isfinite(m):
            continue
        ax.errorbar([m], [y[ci] + off], xerr=[[m - lo], [hi - m]], fmt="o", ms=6.5,
                    lw=1.5, capsize=3, color=col, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.0,
                    label=lbl if ci == 0 else None)
        if sig:
            ax.annotate("*", (m, y[ci] + off), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=12, color=col, fontweight="bold")
ax.axvline(0, color="#444", lw=1.1, ls="--", zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([c for c, _ in CLAIMS], fontsize=8.3)
ax.set_xlabel("Δ Pearson r  (95 % CI, cluster bootstrap over proteins)", fontsize=8.6)
ax.set_title("B · Replicated on both corpora — except the last\n(* CI excludes 0)",
             fontsize=10.5, pad=10)
ax.grid(axis="x", alpha=0.3, lw=0.6, color=GRID)
ax.set_axisbelow(True)

ax = axes[2]
PTS = [("base", "concat (levels)"), ("dz", "Δz"), ("diag", "diagonal"),
       ("dz_cw", "diag+contact"), ("cw", "contact levels"),
       ("far", "far shell"), ("onehot", "one-hot")]
for cfg, lbl in PTS:
    x, yy = get(cfg, "tsu_oof"), get(cfg, "fireprot_le500")
    if not (np.isfinite(x) and np.isfinite(yy)):
        continue
    bad = cfg in ("base", "far", "onehot")
    ax.scatter([x], [yy], s=70, color=BAD if bad else GREEN, zorder=3,
               edgecolor="white", linewidth=1.1)
    # the four difference-form points pile up near x=0.79; label them leftwards
    right = x > 0.76
    dy = {"dz": -11, "diag": 4, "dz_cw": 5, "cw": -11}.get(cfg, -3)
    ax.annotate(lbl, (x, yy), xytext=(-8 if right else 6, dy),
                textcoords="offset points", ha="right" if right else "left",
                fontsize=7.6, color=INK)
ax.set_xlim(0.44, 0.86)
ax.set_xlabel("Pearson r, held-out Tsuboyama (in-distribution)", fontsize=8.8)
ax.set_ylabel("Pearson r, FireProt transfer", fontsize=8.8)
ax.set_title("C · In-distribution skill does not imply transfer\n"
             "(red: built on whole-chain pooling)", fontsize=10.5, pad=10)
ax.grid(alpha=0.3, lw=0.6, color=GRID)
ax.set_axisbelow(True)

fig.legend(hA, lA, fontsize=8.4, frameon=False, ncol=2,
           loc="lower center", bbox_to_anchor=(0.5, -0.05))
fig.tight_layout()
fig.savefig(OUT / "01_what_generalizes.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "01_what_generalizes.png")

fig, axes = plt.subplots(1, 3, figsize=(15.2, 4.5))
MET = [("r", "Pearson r"), ("rho", "Spearman ρ"), ("mae", "MAE (flipped)"),
       ("auc_stab", "AUC stabilizing")]

y = np.arange(len(MET))[::-1]
ax = axes[0]
for (stem, ref, cfg), (st, lbl, col), off in zip(
        [(S6_LOC, "dz", "dz_cw"), (FP_LOC, "dz", "dz_cw")], CORPUS, (0.18, -0.18)):
    for mi, (mk, _) in enumerate(MET):
        s = -1.0 if mk == "mae" else 1.0
        m, lo, hi, sig = boot(stem, ref, cfg, mk)
        if not np.isfinite(m):
            continue
        m, lo, hi = s * m, s * lo, s * hi
        ax.errorbar([m], [y[mi] + off], xerr=[[m - min(lo, hi)], [max(lo, hi) - m]],
                    fmt="o", ms=6, lw=1.4, capsize=3, color=col, zorder=3,
                    markeredgecolor="white", markeredgewidth=1.0,
                    label=lbl if mi == 0 else None)
        if sig:
            ax.annotate("*", (m, y[mi] + off), xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=11, color=col, fontweight="bold")
ax.axvline(0, color="#444", lw=1.1, ls="--", zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([n for _, n in MET], fontsize=8.3)
ax.set_xlabel("change vs uniform pooling (95 % CI)", fontsize=8.5)
ax.set_title("A · Item 1 — contact weighting\nsignificant on one corpus, zero on the other",
             fontsize=10.3, pad=10)
ax.legend(fontsize=7.4, frameon=False, loc="upper left")
ax.grid(axis="x", alpha=0.3, lw=0.6, color=GRID)
ax.set_axisbelow(True)

ax = axes[1]
BARS = [("base", "concat\nbaseline"), ("base+bio_nox", "+biophysics\nmarginals"),
        ("base+bio", "+biophysics\n+interactions"), ("bio", "biophysics\nALONE")]
vals = [get(c, "s669", aug=True) for c, _ in BARS]
cols = [ORANGE, PURPLE, PURPLE, BAD]
b = ax.bar(range(len(BARS)), vals, color=cols, width=0.62, zorder=3)
for rect, v in zip(b, vals):
    ax.annotate(f"{v:.3f}", (rect.get_x() + rect.get_width() / 2, v),
                xytext=(0, 3), textcoords="offset points", ha="center",
                fontsize=8.3, color=INK)
ax.axhline(get("base", "s669", aug=True), color=ORANGE, lw=1.0, ls="--", zorder=2)
ax.set_xticks(range(len(BARS)))
ax.set_xticklabels([l for _, l in BARS], fontsize=8)
ax.set_ylabel("Pearson r, S669 transfer", fontsize=8.8)
ax.set_ylim(0, 0.52)
ax.set_title("B · Item 2 — burial + biophysics\nnever exceeds the baseline",
             fontsize=10.3, pad=10)
ax.grid(axis="y", alpha=0.3, lw=0.6, color=GRID)
ax.set_axisbelow(True)

ax = axes[2]
for mi, (mk, _) in enumerate(MET):
    s = -1.0 if mk == "mae" else 1.0
    m, lo, hi, sig = boot(FP_CONS, "cw", "cw+cons", mk)
    if not np.isfinite(m):
        continue
    m, lo, hi = s * m, s * lo, s * hi
    ax.errorbar([m], [y[mi]], xerr=[[m - min(lo, hi)], [max(lo, hi) - m]],
                fmt="o", ms=6.5, lw=1.5, capsize=3,
                color=BAD if sig else GREEN, zorder=3,
                markeredgecolor="white", markeredgewidth=1.0)
ax.axvline(0, color="#444", lw=1.1, ls="--", zorder=1)
ax.set_yticks(y)
ax.set_yticklabels([n for _, n in MET], fontsize=8.3)
ax.set_xlabel("change vs contact-weighted alone (95 % CI)", fontsize=8.5)
ax.set_title("C · Item 3 — MSA conservation\nadds nothing on deep alignments",
             fontsize=10.3, pad=10)
ax.grid(axis="x", alpha=0.3, lw=0.6, color=GRID)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT / "02_additions_that_failed.png", dpi=200, bbox_inches="tight")
plt.close(fig)
print("wrote", OUT / "02_additions_that_failed.png")
