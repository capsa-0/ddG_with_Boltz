"""Figures for 16_aftoolkit_headtohead.

    python results/16_aftoolkit_headtohead/make_figures.py

01 — the S669 head-to-head: both methods scored on the identical 541 variants, the
     paired protein-cluster bootstrap of the difference, and the control showing the
     500-residue cap does not hand us an easier subset.
02 — the leakage audit: which benchmark proteins each training corpus has already
     seen, and what that leaves as a corpus blind to both methods.

Colour identifies the METHOD throughout (AFToolkit blue, this project orange, the
project's in-distribution default purple); leakage panels use the same blue/orange
for "seen in training" vs "blind". Palette re-validated with the dataviz CVD checker
(all six checks pass; worst adjacent pair dE 11.6 deutan / 19.9 normal vision).
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
INK, MUTED, GRID = "#1a1a1a", "#555555", "#c9d1d9"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#8b949e", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

h2h = pd.read_csv(HERE / "headtohead_s669.csv")
boot = pd.read_csv(HERE / "headtohead_s669_bootstrap.csv")
ours = pd.read_csv(HERE / "results_ours.csv")
aft = pd.read_csv(HERE / "aftoolkit_s669_predictions.csv")


def figure1():
    fig = plt.figure(figsize=(12.4, 4.5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.55, 1.0, 0.8], wspace=0.42)

    # --- A: both methods on the identical 541 variants
    ax = fig.add_subplot(gs[0])
    order = ["ours: Boltz-2 zdiag, 128d (best transfer)",
             "ours: Boltz-2 diag + contact-weighted, 256d",
             "ours: Boltz-2 dz, 256d",
             "AFToolkit SVM (AF2 pair+lddt+plddt, 223,611 train)",
             "ours: Boltz-2 concat, 256d (project default)",
             "AFToolkit MLP", "AFToolkit CatBoost"]
    short = ["Boltz-2 zdiag 128d", "Boltz-2 diag+cw 256d", "Boltz-2 dz 256d",
             "AFToolkit SVM", "Boltz-2 concat 256d\n(project default)",
             "AFToolkit MLP", "AFToolkit CatBoost"]
    d = h2h.set_index("model").loc[order]
    y = np.arange(len(order))[::-1]
    cols = [ORANGE if m.startswith("ours") else BLUE for m in order]
    cols[order.index("ours: Boltz-2 concat, 256d (project default)")] = PURPLE
    ax.barh(y + 0.19, d.rho, 0.34, color=cols, edgecolor="white", linewidth=0.6)
    ax.barh(y - 0.19, d.r, 0.34, color=cols, alpha=0.5, edgecolor="white", linewidth=0.6)
    for yy, (rho, r) in zip(y, zip(d.rho, d.r)):
        ax.text(rho + .008, yy + .19, f"{rho:.3f}", va="center", fontsize=8, color=INK)
        ax.text(r + .008, yy - .19, f"{r:.3f}", va="center", fontsize=8, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(short, fontsize=8.2)
    ax.set_xlabel("correlation with experimental ΔΔG"); ax.set_xlim(0, 0.72)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("A · S669, the same 541 variants / 62 proteins\n"
                 "solid = Spearman ρ, pale = Pearson r", fontsize=9.5, loc="left")

    # --- B: paired bootstrap difference
    ax = fig.add_subplot(gs[1])
    lab = {"zdiag 128d": "Boltz-2 zdiag 128d", "dz 256d": "Boltz-2 dz 256d",
           "concat 256d": "Boltz-2 concat 256d\n(project default)"}
    yy = np.arange(len(boot))[::-1] * 1.0
    for i, (y0, r) in enumerate(zip(yy, boot.itertuples())):
        for off, m, c, a in ((+.17, "rho", ORANGE, 1.0), (-.17, "r", ORANGE, 0.5)):
            lo, hi = r.d_rho_lo if m == "rho" else r.d_r_lo, r.d_rho_hi if m == "rho" else r.d_r_hi
            mid = r.d_rho if m == "rho" else r.d_r
            col = c if mid > 0 else PURPLE
            ax.plot([lo, hi], [y0 + off] * 2, color=col, lw=2.0, alpha=a,
                    solid_capstyle="round")
            ax.plot(mid, y0 + off, "o", ms=6, color=col, alpha=a,
                    markeredgecolor="white", markeredgewidth=1.0)
    ax.axvline(0, color=INK, lw=1.0, ls="--", alpha=.55)
    ax.set_yticks(yy); ax.set_yticklabels([lab[b] for b in boot.ours], fontsize=8.2)
    ax.set_xlabel("Δ correlation vs AFToolkit SVM")
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.set_title("B · paired difference, same variants\n"
                 "4,000-resample protein bootstrap, 95 % CI", fontsize=9.5, loc="left")
    ax.set_ylim(-0.55, len(boot) - 0.45)
    ax.legend(handles=[plt.Line2D([], [], color=ORANGE, lw=2.4, marker="o", ms=5.5),
                       plt.Line2D([], [], color=ORANGE, lw=2.4, marker="o", ms=5.5, alpha=.5)],
              labels=["Spearman ρ", "Pearson r"], loc="lower left", frameon=False,
              fontsize=7.8, handlelength=1.6)
    ax.text(1.0, -0.20, "right of 0 = this project ahead", transform=ax.transAxes,
            fontsize=7.5, color=MUTED, ha="right")

    # --- C: is our 500-aa subset the easy half?
    ax = fig.add_subplot(gs[2])
    kept, excl = aft[aft.seqlen <= 500], aft[aft.seqlen > 500]
    from scipy.stats import spearmanr
    vals = [spearmanr(kept.ddg, -kept.aft_svm).statistic,
            spearmanr(excl.ddg, -excl.aft_svm).statistic]
    b = ax.bar([0, 1], vals, 0.55, color=[BLUE, "#8fb8dc"], edgecolor="white", linewidth=0.6)
    for x, v, n in zip([0, 1], vals, [len(kept), len(excl)]):
        ax.text(x, v + .012, f"{v:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(x, .02, f"n={n}", ha="center", fontsize=8, color="white")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["≤500 aa\n(scored here)", ">500 aa\n(excluded)"],
                                              fontsize=8.2)
    ax.set_ylabel("AFToolkit SVM, Spearman ρ"); ax.set_ylim(0, 0.66)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.set_title("C · the cap is not a\nfavourable subset", fontsize=9.5, loc="left")

    fig.savefig(OUT / "01_s669_headtohead.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def figure2():
    ov = pd.read_csv(HERE / "fireprot_aftoolkit_train_overlap.csv")
    leak = pd.read_csv(HERE.parents[0] / "09_external_benchmarks/homology/s669_leakage.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 3.7),
                             gridspec_kw=dict(width_ratios=[1.15, 1.0], wspace=0.32))

    # --- A: who has already seen which benchmark protein
    ax = axes[0]
    rows = [("S669\nvs this project's training corpus", int(leak.leaky_tsu_30.sum()), len(leak)),
            ("FireProt ≤500\nvs this project's training corpus", 8, 138),
            ("FireProt ≤500\nvs AFToolkit's training corpus",
             int(ov.drop_duplicates('wt_id').protein_in_aft_train.sum()),
             ov.wt_id.nunique())]
    y = np.arange(len(rows))[::-1]
    for y0, (lab, seen, tot) in zip(y, rows):
        ax.barh(y0, tot - seen, 0.42, color=ORANGE, edgecolor="white", linewidth=0.8)
        ax.barh(y0, seen, 0.42, left=tot - seen, color=BLUE, edgecolor="white", linewidth=0.8)
        ax.text(tot + 2, y0, f"{seen} of {tot} seen", va="center", fontsize=8.4, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.2)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_xlabel("benchmark proteins"); ax.set_xlim(0, 168)
    ax.xaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=ORANGE),
                       plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["blind (never in training)", "already in the training set"],
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2,
              frameon=False, fontsize=8)
    ax.set_title("A · leakage audit — proteins each method has already trained on\n"
                 "MMseqs2 30 % identity for this project; PDB identity for AFToolkit",
                 fontsize=9.5, loc="left")

    # --- B: our FireProt score, full vs blind-to-both
    ax = axes[1]
    cfgs = ["diag", "dz_cw", "dz", "cw", "base"]
    names = ["zdiag 128d", "diag+cw 256d", "dz 256d", "cw 256d", "concat 256d"]
    full = ours[ours.corpus == "fireprot_le500_filtered"].set_index("config")
    blind = ours[ours.corpus == "fireprot_blind_to_both"].set_index("config")
    x = np.arange(len(cfgs))
    ax.bar(x - .19, [full.loc[c, "rho"] for c in cfgs], .34, color=ORANGE,
           edgecolor="white", linewidth=.6, label=f"all 130 proteins (n={int(full.n.iloc[0])})")
    ax.bar(x + .19, [blind.loc[c, "rho"] for c in cfgs], .34, color=GREEN,
           edgecolor="white", linewidth=.6,
           label=f"blind to both methods, 40 proteins (n={int(blind.n.iloc[0])})")
    for xi, c in zip(x, cfgs):
        ax.text(xi - .19, full.loc[c, "rho"] + .01, f"{full.loc[c,'rho']:.2f}",
                ha="center", fontsize=7.6, color=INK)
        ax.text(xi + .19, blind.loc[c, "rho"] + .01, f"{blind.loc[c,'rho']:.2f}",
                ha="center", fontsize=7.6, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8, rotation=18, ha="right")
    ax.set_ylabel("Spearman ρ, this project"); ax.set_ylim(0, 0.82)
    ax.yaxis.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.set_title("B · restricting FireProt to the proteins blind to both methods\n"
                 "barely moves this project's score", fontsize=9.5, loc="left")

    fig.savefig(OUT / "02_leakage_audit.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


figure1()
figure2()
print("wrote", OUT / "01_s669_headtohead.png", "and", OUT / "02_leakage_audit.png")
