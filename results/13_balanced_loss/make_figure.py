"""Figure: what balanced losses do and do not fix."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)
STAB = -0.5

d = pd.read_csv(ROOT / "data/processed/_analysis/balanced_oof.csv")
bs = pd.read_csv(Path(__file__).resolve().parent / "bootstrap.csv")

C = {"mse": "#4F5D5A", "bmc": "#0E6C68", "lds": "#9A5B0C"}
LBL = {"mse": "MSE", "bmc": "Balanced MSE", "lds": "LDS"}

fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))

# --- 1. paired difference vs MSE, with CIs ---
ax = axes[0]
metrics = ["rho", "stab_rho", "auc_stab", "detpr30", "stab_bias", "r", "mae"]
nice = {"rho": "ρ overall", "stab_rho": "ρ stabilizing", "auc_stab": "AUC stabilizing",
        "detpr30": "DetPr@30", "stab_bias": "bias stabilizing", "r": "r overall",
        "mae": "MAE overall"}
piv = {k: bs[bs.loss == k].set_index("metric") for k in C}
ypos = np.arange(len(metrics))[::-1]
for off, k in ((-0.18, "bmc"), (+0.18, "lds")):
    lo = [piv[k].loc[m, "lo95"] - piv["mse"].loc[m, "mean"] for m in metrics]
    hi = [piv[k].loc[m, "hi95"] - piv["mse"].loc[m, "mean"] for m in metrics]
    mid = [piv[k].loc[m, "mean"] - piv["mse"].loc[m, "mean"] for m in metrics]
    ax.errorbar(mid, ypos + off, xerr=[np.array(mid) - np.array(lo),
                                       np.array(hi) - np.array(mid)],
                fmt="o", ms=5, lw=1.4, capsize=3, color=C[k], label=LBL[k])
ax.axvline(0, color="#444", lw=1.1, ls="--")
ax.set_yticks(ypos)
ax.set_yticklabels([nice[m] for m in metrics], fontsize=9)
ax.set_xlabel("change vs MSE  (bootstrap mean, 95 % CI)", fontsize=9)
ax.set_title("Only the stabilizing BIAS moves\n(lower is better for bias and MAE)",
             fontsize=10.5, pad=10)
ax.legend(fontsize=8, frameon=False, loc="lower right")
ax.grid(axis="x", alpha=0.25, lw=0.6)
ax.set_axisbelow(True)

# --- 2. stabilizing region, predicted vs true ---
ax = axes[1]
s = d[d.y < 0.5]
for k in ("mse", "bmc"):
    ax.scatter(s.y, s[f"pred_{k}"], s=7, alpha=0.35, color=C[k], edgecolors="none",
               label=f"{LBL[k]}  (bias {piv[k].loc['stab_bias','mean']:+.2f})")
lim = [-2.2, 0.6]
ax.plot(lim, lim, color="#444", lw=1.2, ls="--", zorder=0)
ax.axvline(STAB, color="#B3261E", lw=1.0, ls=":")
ax.set_xlim(lim); ax.set_ylim([-1.6, 1.6])
ax.set_xlabel("true ΔΔG  kcal/mol", fontsize=9)
ax.set_ylabel("predicted ΔΔG  kcal/mol", fontsize=9)
ax.set_title("Stabilizing region: BMC shifts predictions down…", fontsize=10.5, pad=10)
lg = ax.legend(fontsize=8, frameon=False, loc="upper left")
for h in lg.legend_handles:
    h.set_sizes([26]); h.set_alpha(0.9)
ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)

# --- 3. ...but ranking is unchanged: detection curve ---
ax = axes[2]
y = d.y.to_numpy()
stab = y < STAB
ks = np.arange(10, 401, 10)
for k in ("mse", "bmc", "lds"):
    p = d[f"pred_{k}"].to_numpy()
    order = np.argsort(p)
    prec = [stab[order[:kk]].mean() for kk in ks]
    ax.plot(ks, prec, color=C[k], lw=1.8, label=LBL[k])
ax.axhline(stab.mean(), color="#444", lw=1.0, ls="--",
           label=f"random ({stab.mean():.3f})")
ax.set_xlabel("top-K most-stabilizing predictions", fontsize=9)
ax.set_ylabel("precision (fraction truly stabilizing)", fontsize=9)
ax.set_title("…without finding more of them", fontsize=10.5, pad=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)

fig.suptitle("Balanced losses on the stabilizing tail — held-out Tsuboyama "
             "(12,359 out-of-fold, 412 proteins)", fontsize=11.5, y=1.0)
fig.tight_layout()
fig.savefig(OUT / "01_balanced_loss.png", dpi=190, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT/'01_balanced_loss.png'}")
