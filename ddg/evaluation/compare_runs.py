"""
Module: compare_runs
Description: Compare two benchmark runs (each a `benchmark_summary.csv` produced by
ddg.evaluation) holdout-by-holdout. Writes a merged comparison table and a grouped
bar chart of pooled Pearson r. Used e.g. for the MSA vs. no-MSA ablation, but works
for any two runs on the same holdout definitions.

CLI
---
    python -m ddg.evaluation.compare_runs \
        --a <benchmark_summary.csv> --label-a MSA \
        --b <benchmark_summary.csv> --label-b no-MSA \
        --out results/04_no_msa_ablation
"""

import argparse
from pathlib import Path

import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Canonical holdout order (single-axis, hardest last-ish).
HOLDOUT_ORDER = ["random", "protein", "cluster", "denovo", "substitution",
                 "source_residue", "target_residue", "chemistry"]


def build_comparison(a_csv: str, b_csv: str,
                     label_a: str, label_b: str) -> pd.DataFrame:
    """Merge two benchmark summaries into one comparison table (Δ = b − a)."""
    a = pd.read_csv(a_csv).set_index("holdout")
    b = pd.read_csv(b_csv).set_index("holdout")
    order = [h for h in HOLDOUT_ORDER if h in a.index or h in b.index]
    order += [h for h in a.index.union(b.index) if h not in order]

    rows = []
    for h in order:
        row = {"holdout": h}
        for lbl, df in ((label_a, a), (label_b, b)):
            for metric in ("pooled_pearson", "pooled_rmse", "pooled_mae"):
                row[f"{lbl}_{metric.split('_')[1]}"] = (
                    float(df.loc[h, metric]) if h in df.index else float("nan"))
        row["delta_pearson"] = row.get(f"{label_b}_pearson", float("nan")) - \
            row.get(f"{label_a}_pearson", float("nan"))
        rows.append(row)
    return pd.DataFrame(rows)


def _plot(cmp: pd.DataFrame, label_a: str, label_b: str, path: Path) -> None:
    import numpy as np
    d = cmp.dropna(subset=[f"{label_a}_pearson", f"{label_b}_pearson"])
    x = np.arange(len(d))
    w = 0.38
    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.bar(x - w / 2, d[f"{label_a}_pearson"], w, label=label_a, color="#4C72B0")
    ax.bar(x + w / 2, d[f"{label_b}_pearson"], w, label=label_b, color="#C44E52")
    for xi, (va, vb) in enumerate(zip(d[f"{label_a}_pearson"], d[f"{label_b}_pearson"])):
        ax.annotate(f"{va:.2f}", (xi - w / 2, va), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{vb:.2f}", (xi + w / 2, vb), ha="center", va="bottom", fontsize=7)
        ax.annotate(f"{vb - va:+.02f}", (xi, max(va, vb) + 0.03), ha="center",
                    fontsize=7, color="0.3")
    ax.set_xticks(x)
    ax.set_xticklabels(d["holdout"], rotation=30, ha="right")
    ax.set_ylabel("Pooled Pearson r")
    ax.set_ylim(0, max(d[[f"{label_a}_pearson", f"{label_b}_pearson"]].max()) + 0.12)
    ax.set_title(f"ΔΔG holdout performance: {label_a} vs. {label_b} (raw-Δz, HGB)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare two benchmark runs")
    ap.add_argument("--a", required=True, help="benchmark_summary.csv (baseline)")
    ap.add_argument("--b", required=True, help="benchmark_summary.csv (comparison)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out", required=True, help="output directory")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmp = build_comparison(args.a, args.b, args.label_a, args.label_b)
    cmp.to_csv(out / "comparison.csv", index=False)
    _plot(cmp, args.label_a, args.label_b, out / "comparison.png")
    print(cmp.to_string(index=False))


if __name__ == "__main__":
    main()
