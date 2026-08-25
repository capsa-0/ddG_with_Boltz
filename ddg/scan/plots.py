"""
Module: plots
Description: Figures for a full mutational scan.

Three views of the same table, answering different questions:
  01 heatmap        — position x residue: which substitutions at which sites.
  02 position profile — per-position mean ΔΔG: which *regions* are sensitive.
  03 regime spread  — how much the three training regimes agree, i.e. where the
                      prediction is extrapolating past every corpus.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ddg.scan.mutations import AA_ORDER  # noqa: E402

logger = logging.getLogger(__name__)

# Positions per heatmap row-band. A 400-residue protein on one axis is unreadable,
# so the heatmap is broken into stacked bands of this width.
BAND = 100
# Diverging map centred on 0: blue = stabilizing, red = destabilizing.
CMAP = "RdBu_r"


# Positions are always the *reported* numbering written by ddg.scan.predict
# (scan.first_residue applied), so figures and the external tables they get
# compared against carry the same residue numbers.
POSITION = "position"


def _matrix_for(predictions, column, pos_col):
    """(positions, 20 x N array) for a prediction column, in AA_ORDER row order."""
    pivot = (predictions.pivot(index=pos_col, columns="mut_aa", values=column)
             .reindex(columns=list(AA_ORDER)).sort_index())
    return pivot.index.to_numpy(), pivot.to_numpy(dtype=float).T


def plot_heatmap(predictions, path, column="ddg_mean",
                 title="Predicted ΔΔG — every point mutation"):
    """Position x residue heatmap, split into stacked bands of BAND positions.

    Cells with no value are drawn grey, never white: white is a real ΔΔG of ~0 on a
    diverging map, so leaving gaps unpainted would make "not computed" look like
    "predicted neutral". Grey cells are the wild-type residue's own row (no such
    mutation) and any substitution whose structure was not predicted.
    """
    positions, values = _matrix_for(predictions, column, POSITION)
    limit = float(np.nanmax(np.abs(values))) or 1.0
    cmap = plt.get_cmap(CMAP).copy()
    cmap.set_bad("#B0B0B0")
    # A scan may cover non-contiguous sites (ddg.scan build --positions/--wt-residues);
    # columns are then adjacent on screen but NOT adjacent in sequence, so every column
    # gets its own label rather than a misleading evenly-spaced numeric axis.
    contiguous = bool(np.all(np.diff(positions) == 1)) if len(positions) > 1 else True

    bands = [(i, min(i + BAND, len(positions))) for i in range(0, len(positions), BAND)]
    fig, axes = plt.subplots(len(bands), 1, figsize=(14, 2.6 * len(bands) + 1.2),
                             squeeze=False)
    image = None
    for ax, (start, stop) in zip(axes[:, 0], bands):
        image = ax.imshow(np.ma.masked_invalid(values[:, start:stop]), aspect="auto",
                          cmap=cmap, vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_yticks(range(len(AA_ORDER)))
        ax.set_yticklabels(list(AA_ORDER), fontsize=7)
        wt = predictions.drop_duplicates(POSITION).set_index(POSITION)["wt_aa"]
        if contiguous:
            ticks = np.arange(start, stop, 10)
            ax.set_xticks(ticks - start)
            ax.set_xticklabels(positions[ticks], fontsize=7)
        else:
            ax.set_xticks(np.arange(stop - start))
            ax.set_xticklabels(
                [f"{wt.loc[p]}{p}" for p in positions[start:stop]],
                fontsize=6, rotation=90)
        ax.set_ylabel("mutant residue", fontsize=8)
        # Outline the wild-type residue's own cell so it reads as "not a mutation"
        # rather than as missing data.
        for offset, position in enumerate(positions[start:stop]):
            row = AA_ORDER.find(wt.loc[position])
            if row >= 0:
                ax.add_patch(plt.Rectangle((offset - .5, row - .5), 1, 1, fill=False,
                                           edgecolor="#404040", linewidth=.8))
    axes[-1, 0].set_xlabel(
        "residue" if contiguous else
        "scanned residue  (NOT contiguous — 38 selected sites, adjacent on screen only)",
        fontsize=9)
    fig.suptitle(title, fontsize=12)
    from matplotlib.patches import Patch
    axes[0, 0].legend(handles=[
        Patch(facecolor="#B0B0B0", label="no value (wild-type cell, or not computed)"),
        Patch(facecolor="white", edgecolor="#404040", label="wild-type residue"),
    ], fontsize=7, loc="upper left", bbox_to_anchor=(0, 1.28), ncol=2, framealpha=.9)
    fig.colorbar(image, ax=axes[:, 0].tolist(), fraction=0.015, pad=0.012,
                 label="predicted ΔΔG (kcal/mol) — positive = destabilizing")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_position_profile(predictions, path, regimes=("ddg_mean",),
                          title="Per-position mean predicted ΔΔG"):
    """Mean ΔΔG over the 19 substitutions at each position, per regime."""
    fig, ax = plt.subplots(figsize=(14, 4.2))
    for column in regimes:
        profile = predictions.groupby(POSITION)[column].mean()
        ax.plot(profile.index, profile.to_numpy(), lw=1.0,
                label=column.replace("ddg_", ""),
                alpha=0.9 if column == "ddg_mean" else 0.55)
    overall = predictions["ddg_mean"].mean()
    ax.axhline(overall, ls=":", lw=1, color="0.4",
               label=f"scan mean ({overall:+.2f})")
    ax.set_xlabel("residue")
    ax.set_ylabel("mean predicted ΔΔG (kcal/mol)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_regime_spread(predictions, path, regimes=("A_tsuboyama", "B_fireprot",
                                                   "D_finetuned")):
    """Distribution per regime + how far the regimes disagree per mutation."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.2))
    for regime in regimes:
        left.hist(predictions[f"ddg_{regime}"], bins=60, histtype="step", lw=1.4,
                  label=regime)
    left.set_xlabel("predicted ΔΔG (kcal/mol)")
    left.set_ylabel("mutations")
    left.set_title("Prediction distribution by training regime")
    left.legend(fontsize=8)
    left.grid(True, alpha=0.3)

    right.scatter(predictions["ddg_mean"], predictions["ddg_regime_sd"],
                  s=4, alpha=0.25, edgecolors="none", color="#4C72B0")
    right.set_xlabel("mean predicted ΔΔG across regimes (kcal/mol)")
    right.set_ylabel("SD across regimes (kcal/mol)")
    right.set_title("Where the regimes disagree\n(high SD = outside every corpus)")
    right.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def make_all(predictions, out_dir) -> list:
    """Write every scan figure into ``out_dir``; returns the paths written."""
    from pathlib import Path
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wt_id = predictions["wt_id"].iloc[0]
    paths = []

    path = out_dir / "01_heatmap_mean.png"
    plot_heatmap(predictions, path,
                 title=f"{wt_id} — predicted ΔΔG for every point mutation "
                       f"(mean of regimes A/B/D)")
    paths.append(path)

    for regime in ("A_tsuboyama", "B_fireprot", "D_finetuned"):
        path = out_dir / f"01_heatmap_{regime}.png"
        plot_heatmap(predictions, path, column=f"ddg_{regime}",
                     title=f"{wt_id} — predicted ΔΔG (regime {regime})")
        paths.append(path)

    path = out_dir / "02_position_profile.png"
    plot_position_profile(
        predictions, path,
        regimes=("ddg_mean", "ddg_A_tsuboyama", "ddg_B_fireprot", "ddg_D_finetuned"),
        title=f"{wt_id} — per-position mean predicted ΔΔG (19 substitutions each)")
    paths.append(path)

    path = out_dir / "03_regime_spread.png"
    plot_regime_spread(predictions, path)
    paths.append(path)

    logger.info("scan figures -> %s", out_dir)
    return paths
