"""Summary figure for results/09: pooled Pearson r, full vs homology-filtered (25%),
per regime and benchmark. Reads results.csv, writes figures/01_pooled_r_full_vs_filtered.png.

    python results/09_external_benchmarks/make_figures.py
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent
res = pd.read_csv(OUT / "results.csv")
regs = ["A_tsu_only", "B_fp_only", "D_finetuned"]
lbl = {"A_tsu_only": "A: Tsuboyama", "B_fp_only": "B: FireProt", "D_finetuned": "D: Tsu→FT FP"}


def val(d, r, sub, col):
    row = d[(d.regime == r) & (d.subset == sub)]
    return row[col].iloc[0]


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
for ax, bench, title in zip(axes, ("s669", "ssym"),
                            ("S669 (541 var / 62 prot)", "Ssym (337 var / 13 prot)")):
    d = res[res.benchmark == bench]
    x = range(len(regs)); w = 0.35
    # red bar = each regime's OWN homology filter (A is leak-free -> filt == full).
    full = [val(d, r, "full", "pearson") for r in regs]
    filt = [val(d, r, "filt25", "pearson") for r in regs]
    nf = [int(val(d, r, "full", "n")) for r in regs]
    nfi = [int(val(d, r, "filt25", "n")) for r in regs]
    ax.bar([i - w/2 for i in x], full, w, label="full", color="#4C72B0")
    ax.bar([i + w/2 for i in x], filt, w, label="homology-filtered (25%)", color="#C44E52")
    for i, (vf, vc, a, b) in enumerate(zip(full, filt, nf, nfi)):
        ax.text(i - w/2, vf + 0.012, f"{vf:.2f}\nn={a}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w/2, vc + 0.012, f"{vc:.2f}\nn={b}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(list(x)); ax.set_xticklabels([lbl[r] for r in regs], fontsize=8)
    ax.set_ylim(0, 1.02); ax.set_ylabel("pooled Pearson r"); ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8, loc="upper left")
fig.suptitle("ΔΔG prediction on external blind benchmarks — full vs homology-filtered\n"
             "A (Tsuboyama) has 0 benchmark overlap → bars equal; B/D drop when FireProt "
             "homologs removed = leakage", fontsize=10)
fig.tight_layout()
fig.savefig(OUT / "figures" / "01_pooled_r_full_vs_filtered.png", dpi=150)
print("wrote figures/01_pooled_r_full_vs_filtered.png")


# --------------------------------------------------------------------------- #
# Figure 2 — what the estimator correction changed, and the antisymmetry bias.
#
# Both panels exist because of the defect found in results/14: run_benchmarks.py
# used `max_iter=250, early_stopping=False`. Since max_iter counts EPOCHS, the
# regime with the most data over-trained hardest, which biased the very comparison
# the experiment exists to make. Panel (a) shows the correction is not uniform;
# panel (b) shows the residual antisymmetry bias, which is regime-dependent and was
# understated in the folder's earlier write-up.
# --------------------------------------------------------------------------- #
pre = pd.read_csv(OUT / "results_pre-correction.csv")
anti = pd.read_csv(OUT / "ssym_antisymmetry.csv")

fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11.4, 4.4),
                                gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.28))

rows = [("s669", "full"), ("s669", "common25"), ("ssym", "full")]
lab = {"s669": "S669", "ssym": "Ssym"}
xs, w = range(len(rows) * 3), 0.8
pos, before, after, ticks = [], [], [], []
for k, (bench, sub) in enumerate(rows):
    for j, r in enumerate(regs):
        i = k * 4 + j
        pos.append(i)
        before.append(val(pre[pre.benchmark == bench], r, sub, "pearson"))
        after.append(val(res[res.benchmark == bench], r, sub, "pearson"))
        ticks.append((i, r[0]))
axa.bar([p - 0.19 for p in pos], before, 0.36, label="before (defective estimator)",
        color="#bdbdbd")
axa.bar([p + 0.19 for p in pos], after, 0.36, label="after (project default)",
        color="#4C72B0")
for p, b, a in zip(pos, before, after):
    axa.annotate("", xy=(p + 0.19, a), xytext=(p - 0.19, b),
                 arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.1))
    axa.text(p, max(a, b) + 0.03, f"{a - b:+.2f}", ha="center", fontsize=7.5,
             color="#C44E52")
axa.set_xticks([t[0] for t in ticks])
axa.set_xticklabels([t[1] for t in ticks], fontsize=8)
for k, (bench, sub) in enumerate(rows):
    axa.text(k * 4 + 1, -0.13, f"{lab[bench]} · {sub}", ha="center", fontsize=8.5,
             transform=axa.get_xaxis_transform())
axa.set_ylim(0, 1.05)
axa.set_ylabel("pooled Pearson r")
axa.set_title("a  the correction is not uniform — regime A gains most", fontsize=10)
axa.grid(True, axis="y", alpha=0.3)
axa.legend(fontsize=8, loc="upper left")

cols = {"A_tsu_only": "#4C72B0", "B_fp_only": "#55A868", "D_finetuned": "#C44E52"}
axb.barh(range(3), [float(anti[anti.regime == r].bias_mean.iloc[0]) for r in regs],
         0.55, color=[cols[r] for r in regs],
         xerr=[float(anti[anti.regime == r].bias_sd.iloc[0]) for r in regs],
         error_kw=dict(ecolor="#333", lw=1.1, capsize=4))
axb.axvline(0, color="#333", lw=0.9)
axb.set_yticks(range(3))
axb.set_yticklabels([lbl[r] for r in regs], fontsize=8.5)
axb.invert_yaxis()
axb.set_xlabel("mean bias of direct + reverse  (kcal/mol)")
axb.set_title("b  antisymmetry residual — only B is unbiased", fontsize=10)
axb.grid(True, axis="x", alpha=0.3)

fig2.suptitle("The estimator correction, and the residual the model leaves on Ssym's "
              "forward/reverse pairs", fontsize=10.5)
fig2.tight_layout()
fig2.savefig(OUT / "figures" / "02_correction_and_antisymmetry.png", dpi=150)
print("wrote figures/02_correction_and_antisymmetry.png")
