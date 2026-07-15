"""
Module: plots
Description: Figures for the benchmark results.

Everything reads the BenchmarkResults produced by ddg.evaluation.benchmark:
  * holdout_bar        — pooled Pearson per holdout (the headline figure)
  * per_unit_distribution — box/strip of per-protein / per-cluster Pearson
  * prediction_scatter — predicted vs true for one holdout's OOF preds
  * substitution_heatmap — 20x20 source×target Pearson (leave-one-substitution-out)
  * chemistry_bar      — Pearson per chemistry category
"""

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

AA_ORDER = list("AVLIMFWYGSTNQCPKRHDE")  # grouped: hydrophobic, special, polar, +, -


def _save(fig, out_dir, name):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", path)
    return path


def holdout_bar(results, out_dir):
    """Headline: pooled Pearson per holdout, error bars = SD across units."""
    s = results.summary.copy()
    if s.empty:
        return None
    s = s.sort_values("pooled_pearson", ascending=True)
    err = s.get("pearson_sd")
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(s) + 2))
    ax.barh(s["holdout"], s["pooled_pearson"],
            xerr=err if err is not None else None,
            color=sns.color_palette("crest", len(s)), capsize=3)
    # Label inside the bar (left-aligned) so it never collides with the SD whisker.
    for y, v in enumerate(s["pooled_pearson"]):
        ax.text(0.01 if v >= 0 else -0.01, y, f"{v:.3f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=9,
                color="white", fontweight="bold")
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pooled Pearson r (out-of-fold)")
    ax.set_title("ΔΔG generalization by holdout")
    ax.grid(axis="x", ls="--", alpha=0.5)
    return _save(fig, out_dir, "holdout_pearson_bar.png")


def per_unit_distribution(results, out_dir, holdouts=("protein", "cluster")):
    """Box+strip of per-unit Pearson — exposes families where the model dies."""
    frames = []
    for h in holdouts:
        if h in results.per_unit:
            d = results.per_unit[h][["unit", "pearson"]].dropna().copy()
            d["holdout"] = h
            frames.append(d)
    if not frames:
        return None
    data = pd.concat(frames, ignore_index=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.boxplot(data=data, x="holdout", y="pearson", ax=ax,
                color="lightgray", showfliers=False)
    sns.stripplot(data=data, x="holdout", y="pearson", ax=ax,
                  size=3, alpha=0.5, color="steelblue")
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_ylabel("Per-unit Pearson r")
    ax.set_xlabel("")
    ax.set_title("Per-unit ΔΔG accuracy distribution")
    return _save(fig, out_dir, "per_unit_distribution.png")


def prediction_scatter(results, out_dir, holdout="protein"):
    """Predicted vs experimental ΔΔG for one holdout's out-of-fold predictions."""
    if holdout not in results.predictions:
        return None
    d = results.predictions[holdout]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(d["y"], d["pred"], s=8, alpha=0.3, color="steelblue")
    lim = [min(d["y"].min(), d["pred"].min()), max(d["y"].max(), d["pred"].max())]
    ax.plot(lim, lim, "r--", lw=1)
    ax.set_xlabel("Experimental ΔΔG")
    ax.set_ylabel("Predicted ΔΔG")
    r = d["y"].corr(d["pred"])
    ax.set_title(f"{holdout} holdout — r = {r:.3f}, n = {len(d)}")
    ax.set_aspect("equal", "box")
    return _save(fig, out_dir, f"scatter_{holdout}.png")


def substitution_heatmap(results, out_dir, metric="pearson"):
    """20x20 source×target residue matrix from leave-one-substitution-out."""
    if "substitution" not in results.per_unit:
        return None
    d = results.per_unit["substitution"].copy()
    parts = d["unit"].str.split("->", expand=True)
    d["src"], d["dst"] = parts[0], parts[1]
    mat = d.pivot_table(index="src", columns="dst", values=metric, aggfunc="mean")
    mat = mat.reindex(index=AA_ORDER, columns=AA_ORDER)
    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(mat, cmap="vlag", center=0, ax=ax,
                cbar_kws={"label": f"{metric} (leave-one-substitution-out)"})
    ax.set_xlabel("target residue")
    ax.set_ylabel("source residue")
    ax.set_title("Per-substitution ΔΔG accuracy")
    return _save(fig, out_dir, f"substitution_{metric}_heatmap.png")


def chemistry_bar(results, out_dir):
    """Pearson per chemistry category (leave-category-out)."""
    if "chemistry" not in results.per_unit:
        return None
    d = results.per_unit["chemistry"].dropna(subset=["pearson"]).sort_values("pearson")
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(d) + 2))
    ax.barh(d["unit"], d["pearson"], color=sns.color_palette("flare", len(d)))
    for y, (v, n) in enumerate(zip(d["pearson"], d["n"])):
        ax.text(v, y, f" {v:.2f} (n={n})", va="center",
                ha="left" if v >= 0 else "right", fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("Pearson r")
    ax.set_title("ΔΔG accuracy by chemistry-category holdout")
    ax.grid(axis="x", ls="--", alpha=0.5)
    return _save(fig, out_dir, "chemistry_bar.png")


def make_all(results, out_dir):
    """Generate every applicable figure; return the list of written paths."""
    paths = [
        holdout_bar(results, out_dir),
        per_unit_distribution(results, out_dir),
        prediction_scatter(results, out_dir, "protein"),
        prediction_scatter(results, out_dir, "random"),
        substitution_heatmap(results, out_dir, "pearson"),
        chemistry_bar(results, out_dir),
    ]
    return [p for p in paths if p is not None]
