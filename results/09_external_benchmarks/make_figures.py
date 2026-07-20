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
