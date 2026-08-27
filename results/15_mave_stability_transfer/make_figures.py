"""Figures for 15_mave_stability_transfer.

01 — the paired leave-one-protein-out comparison (Hoie's Fig 2B analogue).
     (a) median rho per model; (b) the paired difference Boltz - Rosetta with its
     95% bootstrap CI. Panel (b) is the result: an effect with an interval is a
     dot-and-error-bar, not a bar chart.
02 — per-dataset direct correlation (their Fig 2A analogue), as a dumbbell so the
     Rosetta->Boltz move is one readable segment per dataset rather than two bars
     the eye has to difference.

Colour: categorical slots 1-3 of the validated default palette (blue/orange/aqua),
assigned by identity and in fixed order. Validated with the dataviz six-checks at
--pairs all (worst all-pairs CVD dE 9.2, normal-vision 24.0). The aqua contrast WARN
is discharged the way the rule requires: every value is directly labelled and the
underlying CSVs are committed beside the figures.

Sign: ddG correlates NEGATIVELY with fitness (destabilizing -> low fitness) and GEMME
positively. Figure 02 plots |rho| so the three are comparable in magnitude; the axis
label and caption say so.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"

ROSETTA, BOLTZ, GEMME = "#2a78d6", "#eb6834", "#1baf7a"
SHARED = "#52514e"          # text-secondary: the arms-agnostic models
GRID = "#d8d7d2"
INK, INK2 = "#0b0b0b", "#52514e"
LABEL = {"null_smave": "null (s̃)", "dde_only": "ΔΔE only\n(GEMME)",
         "ddg_only": "ΔΔG only", "ddg_dde": "ΔΔG + ΔΔE",
         "position_context": "position-\ncontext"}
# One-line forms for the forest panel, where a newline would misalign the rows.
LABEL_FLAT = {"null_smave": "null (s̃)", "dde_only": "ΔΔE only (GEMME)",
              "ddg_only": "ΔΔG only", "ddg_dde": "ΔΔG + ΔΔE",
              "position_context": "position-context"}


def _style(ax):
    ax.set_facecolor("#fcfcfb")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9, length=3)


def fig01(summary, boot):
    order = ["null_smave", "dde_only", "ddg_only", "ddg_dde", "position_context"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.4), width_ratios=[1.35, 1],
                               facecolor="#fcfcfb")

    x = np.arange(len(order))
    w = 0.36
    for i, m in enumerate(order):
        rows = summary[summary.model == m]
        shared = rows[rows.arm == "shared"]
        if len(shared):
            a.bar(i, shared.median_spearman.iloc[0], w * 1.6, color=SHARED,
                  edgecolor="#fcfcfb", linewidth=2)
            a.text(i, shared.median_spearman.iloc[0] + .012,
                   f"{shared.median_spearman.iloc[0]:.3f}", ha="center",
                   fontsize=8.5, color=INK)
            continue
        for j, (arm, col) in enumerate((("rosetta", ROSETTA), ("boltz", BOLTZ))):
            r = rows[rows.arm == arm]
            if not len(r):
                continue
            v = r.median_spearman.iloc[0]
            a.bar(i + (j - .5) * w, v, w, color=col, edgecolor="#fcfcfb", linewidth=2)
            a.text(i + (j - .5) * w, v + .012, f"{v:.3f}", ha="center",
                   fontsize=8.5, color=INK)
    a.set_xticks(x)
    a.set_xticklabels([LABEL[m] for m in order], fontsize=9)
    a.set_ylabel("median Spearman ρ across 13 MAVE datasets", fontsize=9.5, color=INK)
    a.set_ylim(0, max(summary.median_spearman) * 1.18)
    a.yaxis.grid(True, color=GRID, linewidth=.7)
    a.set_axisbelow(True)
    _style(a)
    a.set_title("a  leave-one-protein-out, by feature set", fontsize=10.5,
                color=INK, loc="left", pad=10)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (ROSETTA, BOLTZ, SHARED)]
    a.legend(handles, ["Rosetta ΔΔG", "Boltz ΔΔG (ours)", "no ΔΔG term"],
             fontsize=8.5, frameon=False, loc="upper left", labelcolor=INK2)

    bo = boot.set_index("model").loc[[m for m in order if m in set(boot.model)]]
    y = np.arange(len(bo))[::-1]
    b.axvline(0, color=INK2, linewidth=1, zorder=1)
    for yy, (m, r) in zip(y, bo.iterrows()):
        sig = r.ci_lo > 0 or r.ci_hi < 0
        b.plot([r.ci_lo, r.ci_hi], [yy, yy], color=BOLTZ if sig else INK2,
               linewidth=2, solid_capstyle="round", zorder=2)
        b.plot([r.delta], [yy], "o", ms=9, color=BOLTZ if sig else INK2,
               markeredgecolor="#fcfcfb", markeredgewidth=2, zorder=3)
        b.text(r.ci_hi + .008, yy, f"{r.delta:+.3f}  [{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]",
               va="center", fontsize=8.5, color=INK)
    b.set_yticks(y)
    b.set_yticklabels([LABEL_FLAT[m] for m in bo.index], fontsize=9)
    b.set_xlabel("Δ median ρ   (Boltz − Rosetta)", fontsize=9.5, color=INK)
    b.set_xlim(-0.075, 0.205)
    # Keep the three rows visually compact instead of floating in a tall axis, so
    # panel b reads as a companion to panel a rather than a near-empty frame.
    b.set_ylim(-0.75, len(bo) - 0.25)
    b.xaxis.grid(True, color=GRID, linewidth=.7)
    b.set_axisbelow(True)
    _style(b)
    b.set_title("b  paired difference, 95 % CI (protein bootstrap)", fontsize=10.5,
                color=INK, loc="left", pad=10)

    fig.tight_layout()
    fig.savefig(FIG / "01_lopo_paired.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def _short(name: str) -> str:
    """Drop the numeric PRISM prefix and the redundant _DMS/_reordered suffixes."""
    t = name.split("_", 1)[1] if name[:3].isdigit() else name
    for suf in ("_reordered", "_DMS", "_reversed"):
        t = t.replace(suf, "")
    return t.replace("_", " ")


def fig02(l1, regime):
    d = l1.copy()
    d["ros"] = d.rho_rosetta.abs()
    d["blz"] = d[f"rho_boltz_{regime}"].abs()
    d["gem"] = d.rho_gemme.abs()
    d = d.sort_values("ros")
    y = np.arange(len(d))

    fig, ax = plt.subplots(figsize=(9.2, 6.2), facecolor="#fcfcfb")
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.plot([r.ros, r.blz], [yy, yy], color=GRID, linewidth=2, zorder=1,
                solid_capstyle="round")
    ax.scatter(d.gem, y, s=52, color=GEMME, zorder=3, edgecolor="#fcfcfb",
               linewidth=1.6, label="GEMME ΔΔE")
    ax.scatter(d.ros, y, s=52, color=ROSETTA, zorder=4, edgecolor="#fcfcfb",
               linewidth=1.6, label="Rosetta ΔΔG")
    ax.scatter(d.blz, y, s=52, color=BOLTZ, zorder=5, edgecolor="#fcfcfb",
               linewidth=1.6, label="Boltz ΔΔG (ours)")
    # Deltas live in their own right-hand column at a fixed x, so they never
    # collide with a marker whose position varies row to row.
    hi = max(d[["ros", "blz", "gem"]].max())
    xcol = hi * 1.06
    for yy, (_, r) in zip(y, d.iterrows()):
        ax.text(xcol, yy, f"{r.blz - r.ros:+.02f}", va="center", fontsize=8.5,
                color=INK if r.blz > r.ros else INK2)
    ax.text(xcol, len(d) - 0.35, "Δ|ρ|", fontsize=8.5, color=INK2, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels([_short(r.dataset) for _, r in d.iterrows()], fontsize=8.5)
    ax.set_xlabel("|Spearman ρ| against MAVE fitness   "
                  "(ΔΔG anti-correlates, GEMME correlates)", fontsize=9.5, color=INK)
    ax.set_xlim(0, hi * 1.15)
    ax.xaxis.grid(True, color=GRID, linewidth=.7)
    ax.set_axisbelow(True)
    _style(ax)
    # Upper-left: the only quadrant with no markers in it (every dataset's weakest
    # predictor still exceeds |rho| 0.07, and the top rows start past 0.35).
    ax.legend(fontsize=8.5, frameon=False, loc="upper left", labelcolor=INK2,
              bbox_to_anchor=(0.005, 0.99))
    ax.set_title("Direct per-dataset correlation — no model, no fitting\n"
                 "grey segment = the Rosetta→Boltz move",
                 fontsize=10.5, color=INK, loc="left", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "02_per_dataset_direct.png", dpi=200, facecolor="#fcfcfb")
    plt.close(fig)


def main(argv=None) -> int:
    src = Path(argv[0]) if argv else HERE
    FIG.mkdir(exist_ok=True)
    summary = pd.read_csv(src / "layer2_lopo_summary.csv")
    l1 = pd.read_csv(src / "layer1_direct.csv")
    boot = pd.read_csv(src / "bootstrap_protein.csv")
    fig01(summary, boot)
    fig02(l1, "mean")
    print(f"wrote {FIG}/01_lopo_paired.png and 02_per_dataset_direct.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
