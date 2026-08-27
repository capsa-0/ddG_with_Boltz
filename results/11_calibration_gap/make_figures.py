"""Figure 2 for results/11 — the panel that carries the conclusion.

    python results/11_calibration_gap/make_figures.py

Figure 1 (`01_per_protein_error.png`) is produced by `offset_learn.py`. This builds
`02_ceiling_and_sharing.png` from the committed tables only:

  * `split_half.csv`           — the honest offset ceiling on S669 (cross-dataset)
  * `split_half_tsuboyama.csv` — the same ceiling in-distribution
  * `homology_share.csv`       — do homologues share the quantity, or the error on it?

The two panels are the argument. (a) An oracle per-protein offset is worth several
times more across datasets than within one, so the missing term is not a fixed property
the model failed to learn. (b) Homologues share the per-protein mean ΔΔG — it *is* a
fold property — but they do not share the model's error on it. What is left is corpus
and assay context, which no representation of the protein can supply.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

INK, SOFT = "#1d2321", "#5d6a64"
CROSS, WITHIN = "#C25A12", "#00966F"   # cross-dataset vs in-distribution
GREY = "#9aa39e"


def q(df, name, col="mean"):
    return float(df[df.quantity == name][col].iloc[0])


s669 = pd.read_csv(HERE / "split_half.csv")
tsu = pd.read_csv(HERE / "split_half_tsuboyama.csv")
hom = pd.read_csv(HERE / "homology_share.csv")

fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.2, 4.6),
                              gridspec_kw=dict(width_ratios=[1.15, 1], wspace=0.30))
fig.subplots_adjust(left=.075, right=.975, top=.755, bottom=.145)

# ---- (a) the ceiling, cross-dataset vs in-distribution ---------------------
sets = [("S669\ncross-dataset", s669, CROSS), ("Tsuboyama\nin-distribution", tsu, WITHIN)]
x, w = np.arange(2), 0.30
base = [q(d, "baseline_no_offset") for _, d, _ in sets]
hon = [q(d, "offset_from_other_half_honest") for _, d, _ in sets]
ax.bar(x - w / 2, base, w, color=GREY, label="baseline")
for i, (_, _, c) in enumerate(sets):
    ax.bar(x[i] + w / 2, hon[i], w, color=c, label="with an oracle per-protein offset"
           if i == 0 else None)
for i in range(2):
    gain = hon[i] - base[i]
    ax.annotate("", xy=(x[i] + w / 2, hon[i]), xytext=(x[i] - w / 2, base[i]),
                arrowprops=dict(arrowstyle="->", color=INK, lw=1.2))
    ax.text(x[i], max(base[i], hon[i]) + 0.022, f"{gain:+.3f}", ha="center",
            fontsize=11, color=INK, weight="semibold")
    ax.text(x[i] - w / 2, base[i] - 0.045, f"{base[i]:.3f}", ha="center", fontsize=8.5,
            color="white")
    ax.text(x[i] + w / 2, hon[i] - 0.045, f"{hon[i]:.3f}", ha="center", fontsize=8.5,
            color="white")
ax.set_xticks(x)
ax.set_xticklabels([n for n, _, _ in sets], fontsize=9.5)
ax.set_ylim(0, 0.95)
ax.set_ylabel("pooled Pearson r", fontsize=9.5, color=SOFT)
ax.set_title("a  the same correction, two regimes", fontsize=11, color=INK,
             weight="semibold", pad=8)
ax.legend(frameon=False, fontsize=8.6, loc="upper left")
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=SOFT, labelsize=8.5)
ratio = (hon[0] - base[0]) / (hon[1] - base[1])
ax.text(0.5, -0.19, f"the offset is worth {ratio:.0f}× more across datasets than within one",
        transform=ax.transAxes, ha="center", fontsize=9, color=SOFT)

# ---- (b) is it a fold property, or corpus context? -------------------------
sel = hom[hom.grouping == "same base structure"].set_index("quantity")
order = [("mean_ddg", "the protein's\nmean ΔΔG", WITHIN),
         ("offset", "the model's ERROR\non that mean", CROSS)]
y = np.arange(len(order))
vals = [float(sel.loc[k, "pair_r"]) for k, _, _ in order]
errs = [float(sel.loc[k, "pair_r_sd"]) for k, _, _ in order]
ax2.barh(y, vals, 0.45, color=[c for _, _, c in order],
         xerr=errs, error_kw=dict(ecolor=INK, lw=1.2, capsize=5))
for i, v in enumerate(vals):
    ax2.text(v + errs[i] + 0.025, i, f"{v:+.3f}", va="center", fontsize=11,
             color=INK, weight="semibold")
ax2.axvline(0, color=SOFT, lw=0.9)
ax2.set_yticks(y)
ax2.set_yticklabels([n for _, n, _ in order], fontsize=9.5)
ax2.set_ylim(1.6, -0.6)
ax2.set_xlim(-0.12, 0.82)
ax2.set_xlabel("correlation between two constructs of the same base structure",
               fontsize=9.2, color=SOFT)
ax2.set_title("b  shared between homologues?", fontsize=11, color=INK,
              weight="semibold", pad=8)
for sp in ("top", "right"):
    ax2.spines[sp].set_visible(False)
ax2.tick_params(colors=SOFT, labelsize=8.5)

fig.text(.075, .935, "The missing per-protein term is corpus context, not a property of the protein",
         fontsize=14.5, color=INK, weight="semibold")
fig.text(.075, .885,
         "(a) A perfect per-protein offset buys far more when train and test come from different "
         "corpora than when they do not — so the model is",
         fontsize=9.2, color=SOFT)
fig.text(.075, .845,
         "already calibrated in-distribution. (b) Homologues share the quantity itself but not the "
         "model's error on it, so no representation of the protein can supply it.",
         fontsize=9.2, color=SOFT)

out = FIGS / "02_ceiling_and_sharing.png"
fig.savefig(out, dpi=170)
plt.close(fig)
print(f"wrote {out}")
