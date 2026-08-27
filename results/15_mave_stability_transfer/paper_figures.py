"""
Reproduce Høie et al. 2022 Figure 1 with our ΔΔG in Rosetta's place.

Their Figure 1 is the paper's conceptual centrepiece: every variant placed on a
2-D "stability–conservation landscape" (ΔΔG on x, GEMME ΔΔE on y), coloured by
experimental fitness (1A), then discretised into a 4x4 sector grid reporting what
fraction of each sector is loss-of-function (1B). It is the figure behind their
claim that high-ΔΔG *and* high-ΔΔE variants are almost always dead, while
low/low variants are almost always fine.

Reproducing it with our ΔΔG asks something the Spearman numbers in `score.py`
cannot: does our number place variants in the *same mechanistic picture*, or does
it merely correlate?

Three things this script establishes, in order:

1. **The reproduction is faithful.** With Rosetta's own ΔΔG on our 13-dataset
   Tier-1 subset, the two corner sectors the paper quotes come back at 84 % / 96 %
   against their published 81 % / 93 % (whole 39-dataset set). So the harness,
   the tertile split and the thresholds are right.

2. **Our ΔΔG reproduces the landscape, but on a compressed axis.** Its sd is
   0.97 kcal/mol against Rosetta's 2.14, so the paper's absolute thresholds
   (2 / 3 / 4.5 kcal/mol) leave the top ΔΔG column nearly empty. Sectors are
   therefore drawn twice: at the paper's absolute cuts (which shows the
   compression) and at **quantile-matched** cuts (which compares the ordering,
   i.e. what Spearman and the RF actually use).

3. **Where our advantage over Rosetta comes from.** AUC for detecting
   loss-of-function, computed *within* strata of conservation, is the panel that
   interprets the experiment's headline: pooled we beat Rosetta by ~+0.05 AUC,
   but within a ΔΔE stratum only by ~+0.02. Conditioning on conservation removes
   roughly half the gap — the first quantitative signature of the MSA confound
   that `README.md` flags as the open question, and it is obtained with no GPU.

**Sign gotcha, verified here, not assumed.** The `gemme_dde` column of
`data/raw/mave_hoie_le200_labels.csv` comes straight from the PRISM merged files'
`gemme_score_01`, and it is stored in the **opposite orientation to the paper's
ΔΔE**: high = evolutionarily tolerated. It correlates *positively* with fitness
(pooled ρ +0.27), which is why `layer1_direct.csv` shows rho_gemme > 0 while
Rosetta and ours are negative. This script uses ΔΔE = 1 − gemme_dde, and
`_check_orientation` asserts that choice against the paper's published corner
percentages rather than trusting the column name.

Usage
-----
    python results/15_mave_stability_transfer/paper_figures.py
    python results/15_mave_stability_transfer/paper_figures.py --regime A_tsuboyama
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

HERE = Path(__file__).resolve().parent
FIGS = HERE / "figures"
LABELS = ROOT / "data/raw/mave_hoie_le200_labels.csv"
PREDS = ROOT / "data/processed/mave_hoie_le200/mave_ddg_predictions.csv"

# Their Figure 1B cuts. ΔΔG in kcal/mol; ΔΔE on the rank-normalised 0–1 scale.
DDG_CUTS = (2.0, 3.0, 4.5)
DDE_CUTS = (0.25, 0.50, 0.75)
# The two corner percentages the paper quotes in the Results text, for the
# faithfulness check. (low ΔΔE & low ΔΔG -> % high fitness; high & high -> % low.)
PUBLISHED_CORNERS = (81.0, 93.0)

N_BOOT = 600
RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def load(regime: str) -> pd.DataFrame:
    """Joined per-variant table: fitness, both ΔΔG arms, ΔΔE, on shared rows.

    Coverage is matched to Rosetta's availability, exactly as `score.py` does —
    otherwise our 100 %-coverage scan would be compared on a different row set.
    """
    lab = pd.read_csv(LABELS)
    pred = pd.read_csv(PREDS).rename(columns={"wt_id": "protein",
                                              "mutation": "variant"})
    col = f"ddg_{regime}"
    if col not in pred.columns:
        raise SystemExit(f"{PREDS} has no column {col!r}")
    df = lab.merge(pred[["protein", "variant", col]], on=["protein", "variant"])
    df = df.rename(columns={col: "ddg_boltz"})

    # Paper convention: ΔΔE ~0 conservative, ~1 disruptive. See module docstring.
    df["dde"] = 1.0 - df["gemme_dde"]
    # Their rank normalisation, per dataset, so the 39 assays share one scale.
    df["s"] = df.groupby("dataset")["s_exp"].rank(pct=True)

    df = df.dropna(subset=["dde", "s", "rosetta_ddg", "ddg_boltz"])
    return df.reset_index(drop=True)


def tertile_split(df: pd.DataFrame) -> pd.DataFrame:
    """Their Figure 1B subset: drop the middle tertile of fitness, label the rest."""
    lo, hi = df["s"].quantile(1 / 3), df["s"].quantile(2 / 3)
    sub = df[(df["s"] <= lo) | (df["s"] >= hi)].copy()
    sub["low"] = (sub["s"] <= lo).astype(int)
    return sub


def sector_grid(sub: pd.DataFrame, col: str, ddg_cuts) -> tuple[np.ndarray, np.ndarray]:
    """4x4 grid of % loss-of-function, and the cell counts. Row 0 = lowest ΔΔE."""
    gi = np.digitize(sub[col], ddg_cuts)
    ei = np.digitize(sub["dde"], DDE_CUTS)
    pct = np.full((4, 4), np.nan)
    n = np.zeros((4, 4), int)
    for a in range(4):
        for b in range(4):
            m = (gi == a) & (ei == b)
            n[b, a] = int(m.sum())
            if m.any():
                pct[b, a] = 100.0 * sub.loc[m, "low"].mean()
    return pct, n


def matched_cuts(df: pd.DataFrame) -> list[float]:
    """Our-ΔΔG cuts placed at the same quantiles Rosetta's cuts sit at."""
    qs = [(df["rosetta_ddg"] < t).mean() for t in DDG_CUTS]
    return [float(df["ddg_boltz"].quantile(q)) for q in qs]


def _check_orientation(sub: pd.DataFrame) -> tuple[float, float]:
    """Reproduce the paper's two quoted corner sectors with Rosetta's own ΔΔG."""
    tol = sub[(sub["dde"] < DDE_CUTS[0]) & (sub["rosetta_ddg"] < DDG_CUTS[0])]
    dead = sub[(sub["dde"] > DDE_CUTS[2]) & (sub["rosetta_ddg"] > DDG_CUTS[2])]
    got = (100 * (1 - tol["low"].mean()), 100 * dead["low"].mean())
    if abs(got[0] - PUBLISHED_CORNERS[0]) > 15 or abs(got[1] - PUBLISHED_CORNERS[1]) > 15:
        raise SystemExit(
            f"corner sectors {got[0]:.0f}%/{got[1]:.0f}% are far from the published "
            f"{PUBLISHED_CORNERS[0]:.0f}%/{PUBLISHED_CORNERS[1]:.0f}% — check the ΔΔE "
            f"orientation and the rank normalisation before trusting these figures")
    return got


# --------------------------------------------------------------------------- #
# analysis: does the advantage survive conditioning on conservation?
# --------------------------------------------------------------------------- #
def strata_auc(sub: pd.DataFrame) -> pd.DataFrame:
    """AUC for loss-of-function pooled, within ΔΔE quartiles, and conditional.

    Three kinds of row, and the distinction is the point of the figure:
      * ``pooled``      — one AUC over everything. Conservation is free to help,
                          because a ΔΔG that correlates with ΔΔE inherits its
                          discrimination.
      * ``Q1..Q4``      — one AUC inside each conservation quartile.
      * ``conditional`` — the mean of the four stratum AUCs, i.e. the pooled
                          comparison with conservation held fixed.

    ``conditional`` is computed as **one statistic per bootstrap resample**, not by
    averaging the four per-stratum intervals — averaging CIs is not a CI. Every
    interval is a cluster bootstrap over the 11 proteins (variants within a protein
    are not independent), paired so the shared resample cancels in the difference.
    """
    edges = sub["dde"].quantile([0, .25, .5, .75, 1.]).to_numpy()
    proteins = sub["protein"].unique()

    def one(frame):
        if frame["low"].nunique() < 2:
            return np.nan, np.nan
        return (roc_auc_score(frame["low"], frame["rosetta_ddg"]),
                roc_auc_score(frame["low"], frame["ddg_boltz"]))

    def by_stratum(frame):
        """The four stratum AUCs of one (possibly resampled) frame, averaged."""
        pairs = [one(frame[(frame["dde"] >= edges[i]) & (frame["dde"] <= edges[i + 1])])
                 for i in range(4)]
        pairs = [p for p in pairs if np.isfinite(p[0]) and np.isfinite(p[1])]
        if not pairs:
            return np.nan, np.nan
        return float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs]))

    strata = [("pooled", sub, one), ("conditional", sub, by_stratum)] + [
        (f"Q{i+1}", sub[(sub["dde"] >= edges[i]) & (sub["dde"] <= edges[i + 1])], one)
        for i in range(4)]

    # One resample of proteins, reused by every row, so the rows are comparable.
    picks = [RNG.choice(proteins, size=len(proteins), replace=True) for _ in range(N_BOOT)]
    by_protein = {p: sub[sub["protein"] == p] for p in proteins}

    rows = []
    for name, frame, fn in strata:
        a_r, a_b = fn(frame)
        keep = set(frame.index)
        deltas = []
        for pick in picks:
            res = pd.concat([by_protein[p] for p in pick])
            res = res[res.index.isin(keep)] if name.startswith("Q") else res
            r, b = fn(res)
            if np.isfinite(r) and np.isfinite(b):
                deltas.append(b - r)
        lo, hi = np.percentile(deltas, [2.5, 97.5]) if deltas else (np.nan, np.nan)
        rows.append(dict(stratum=name, n=len(frame), pct_low=100 * frame["low"].mean(),
                         auc_rosetta=a_r, auc_boltz=a_b, delta=a_b - a_r,
                         ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# figures
# --------------------------------------------------------------------------- #
INK = "#1d2321"
SOFT = "#5d6a64"
# Categorical slots, fixed order by identity, validated with the dataviz six-checks
# at --pairs all: worst all-pairs CVD ΔE 15.7 (tritan) / 10.2 (deutan), worst
# normal-vision ΔE 23.8, chroma and contrast pass. Never cycle or reassign these.
ROS = "#C25A12"   # Rosetta ΔΔG
BOL = "#00966F"   # our Boltz ΔΔG
GEM = "#6A51A3"   # GEMME ΔΔE


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c8d0cb")
    ax.tick_params(colors=SOFT, labelsize=8)


def fig_landscape(df: pd.DataFrame, sub: pd.DataFrame, cuts_b, out: Path) -> None:
    """Their Fig 1A (landscape) over their Fig 1B (sectors), both arms."""
    fig = plt.figure(figsize=(12.6, 9.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.06], width_ratios=[1, 1, 1],
                          hspace=.34, wspace=.26, left=.075, right=.94,
                          top=.868, bottom=.075)

    # ---- row 1: the landscape ------------------------------------------------
    samp = df.sample(min(len(df), 12000), random_state=0)
    for k, (col, cuts, name, xlab) in enumerate([
            ("rosetta_ddg", DDG_CUTS, "Rosetta ΔΔG", "ΔΔG (kcal/mol)"),
            ("ddg_boltz", cuts_b, "our ΔΔG (Boltz)", "ΔΔG (kcal/mol)")]):
        ax = fig.add_subplot(gs[0, k])
        sc = ax.scatter(samp[col], samp["dde"], c=samp["s"], cmap="RdYlBu",
                        s=3.2, alpha=.55, linewidths=0, vmin=0, vmax=1, rasterized=True)
        for t in cuts:
            ax.axvline(t, color=INK, lw=.7, ls=(0, (4, 3)), alpha=.55)
        for t in DDE_CUTS:
            ax.axhline(t, color=INK, lw=.7, ls=(0, (4, 3)), alpha=.55)
        lo = np.percentile(df[col], .3)
        hi = np.percentile(df[col], 99.7)
        ax.set_xlim(lo, hi)
        ax.set_ylim(0, 1)
        ax.set_xlabel(xlab, fontsize=9, color=SOFT)
        if k == 0:
            ax.set_ylabel("GEMME ΔΔE  (0 = conservative)", fontsize=9, color=SOFT)
        ax.set_title(name, fontsize=11, color=INK, pad=8, weight="semibold")
        _style(ax)

    cax = fig.add_subplot(gs[0, 2])
    cax.axis("off")
    cb = fig.colorbar(sc, ax=cax, fraction=.16, pad=.02, aspect=14)
    cb.set_label("fitness s$_{exp}$  (1 = wild-type-like)", fontsize=8.5, color=SOFT)
    cb.ax.tick_params(labelsize=8, colors=SOFT)
    cb.outline.set_visible(False)
    cax.text(.5, .18, "Stability–conservation landscape\nreproducing Fig. 1A of\n"
                      "Høie et al. 2022\n\n"
                      f"{len(df):,} variants · 13 MAVE datasets\n"
                      "coverage matched to Rosetta",
             ha="center", va="center", fontsize=8.6, color=SOFT,
             transform=cax.transAxes, linespacing=1.6)

    # ---- row 2: the sector grids --------------------------------------------
    g_r, n_r = sector_grid(sub, "rosetta_ddg", DDG_CUTS)
    g_b, n_b = sector_grid(sub, "ddg_boltz", cuts_b)
    diff = g_b - g_r

    def draw(ax, g, n, title, sub_t, cmap, norm, fmt="{:.0f}%", min_n=0):
        # Cells below `min_n` are drawn blank and greyed: a sector holding 4
        # variants produces the loudest number on the panel and means nothing.
        shown = np.where(n >= min_n, g, np.nan) if min_n else g
        ax.imshow(shown, cmap=cmap, norm=norm, origin="lower", aspect="auto")
        for b in range(4):
            for a in range(4):
                if not n[b, a]:
                    ax.text(a, b, "—", ha="center", va="center", color=SOFT, fontsize=10)
                    continue
                v = g[b, a]
                weak = n[b, a] < min_n
                dark = (not weak) and (norm(v) > .62 or norm(v) < .2)
                col = "#a8b2ac" if weak else ("white" if dark else INK)
                ax.text(a, b + .10, fmt.format(v), ha="center", va="center",
                        color=col, fontsize=9 if weak else 10.5,
                        style="italic" if weak else "normal",
                        weight="normal" if weak else "semibold")
                ax.text(a, b - .20, f"n={n[b, a]:,}".replace(",", "."), ha="center",
                        va="center", fontsize=7,
                        color="#a8b2ac" if weak else ("white" if dark else SOFT))
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_yticklabels(["<.25", ".25–.50", ".50–.75", ">.75"])
        ax.set_title(title, fontsize=10.5, color=INK, pad=16, weight="semibold")
        ax.text(.5, 1.045, sub_t, transform=ax.transAxes, ha="center",
                fontsize=8.2, color=SOFT)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(colors=SOFT, labelsize=8, length=0)

    norm_p = plt.Normalize(10, 100)
    ax = fig.add_subplot(gs[1, 0])
    draw(ax, g_r, n_r, "Rosetta", "the paper's cuts: 2 · 3 · 4.5 kcal/mol", "RdYlBu_r", norm_p)
    ax.set_ylabel("GEMME ΔΔE", fontsize=9, color=SOFT)
    ax.set_xticklabels([f"<{DDG_CUTS[0]:g}", f"{DDG_CUTS[0]:g}–{DDG_CUTS[1]:g}",
                        f"{DDG_CUTS[1]:g}–{DDG_CUTS[2]:g}", f">{DDG_CUTS[2]:g}"])
    ax.set_xlabel("ΔΔG (kcal/mol)", fontsize=9, color=SOFT)

    ax = fig.add_subplot(gs[1, 1])
    draw(ax, g_b, n_b, "our ΔΔG",
         f"quantile-matched cuts: {cuts_b[0]:.2f} · {cuts_b[1]:.2f} · {cuts_b[2]:.2f}",
         "RdYlBu_r", norm_p)
    ax.set_xticklabels([f"<{cuts_b[0]:.1f}", f"{cuts_b[0]:.1f}–{cuts_b[1]:.1f}",
                        f"{cuts_b[1]:.1f}–{cuts_b[2]:.1f}", f">{cuts_b[2]:.1f}"])
    ax.set_xlabel("ΔΔG (kcal/mol)", fontsize=9, color=SOFT)

    ax = fig.add_subplot(gs[1, 2])
    draw(ax, diff, np.minimum(n_r, n_b), "difference",
         "percentage points, ours − Rosetta  ·  greyed: n < 50", "PuOr_r",
         TwoSlopeNorm(vcenter=0, vmin=-25, vmax=25), fmt="{:+.0f}", min_n=50)
    ax.set_xticklabels(["low", "", "", "high"])
    ax.set_xlabel("ΔΔG (sector)", fontsize=9, color=SOFT)

    fig.text(.075, .958, "The stability–conservation landscape, with our ΔΔG in Rosetta's place",
             fontsize=15.5, color=INK, weight="semibold")
    fig.text(.075, .932,
             "Reproducing Figure 1 of Høie et al. 2022 (Cell Reports 38:110207). Below: % of variants that lose "
             "function, per sector",
             fontsize=9.4, color=SOFT)
    fig.text(.075, .911,
             "(extreme fitness tertiles; the middle third is excluded, as in the paper). The sectors do not hold the "
             "same variants in both arms.",
             fontsize=9.4, color=SOFT)
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_strata(tab: pd.DataFrame, out: Path) -> None:
    """Does the advantage over Rosetta survive conditioning on conservation?"""
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.8),
                                  gridspec_kw=dict(width_ratios=[1.3, 1], wspace=.30))
    fig.subplots_adjust(left=.070, right=.975, top=.755, bottom=.145)

    rows = tab[tab.stratum.str.startswith("Q")].reset_index(drop=True)
    pooled = tab[tab.stratum == "pooled"].iloc[0]
    cond = tab[tab.stratum == "conditional"].iloc[0]
    x = np.arange(len(rows))
    w = .36
    ax.bar(x - w / 2, rows.auc_rosetta, w, color=ROS, label="Rosetta ΔΔG")
    ax.bar(x + w / 2, rows.auc_boltz, w, color=BOL, label="our ΔΔG")
    for i, r in rows.iterrows():
        ax.text(i, max(r.auc_rosetta, r.auc_boltz) + .012, f"{r.delta:+.3f}",
                ha="center", fontsize=8.5, color=INK, weight="semibold")
    ax.axhline(.5, color=SOFT, lw=.8, ls=(0, (3, 3)))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Q{i+1}\n{r.pct_low:.0f} % dead" for i, r in rows.iterrows()],
                       fontsize=8.6)
    ax.set_ylim(.45, .80)
    ax.set_ylabel("AUC — detecting loss of function", fontsize=9.5, color=SOFT)
    ax.set_xlabel("ΔΔE conservation quartile   (Q1 = evolutionarily tolerated  →  Q4 = constrained)",
                  fontsize=9, color=SOFT)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    _style(ax)

    ys = [0, 1]
    vals = [pooled.delta, cond.delta]
    los = [pooled.ci_lo, cond.ci_lo]
    his = [pooled.ci_hi, cond.ci_hi]
    ax2.barh(ys, vals, .42, color=[BOL, "#8fb9ab"])
    ax2.errorbar(vals, ys, xerr=[np.array(vals) - np.array(los),
                                 np.array(his) - np.array(vals)],
                 fmt="none", ecolor=INK, lw=1.3, capsize=5)
    for y, v, lo, hi in zip(ys, vals, los, his):
        ax2.text(hi + .004, y, f"{v:+.3f}", va="center", fontsize=10.5,
                 color=INK, weight="semibold")
        ax2.text(hi + .004, y + .26, f"[{lo:+.3f} · {hi:+.3f}]",
                 va="center", fontsize=8.4, color=SOFT)
    ax2.axvline(0, color=SOFT, lw=.9)
    ax2.set_yticks(ys)
    ax2.set_yticklabels(["pooled\n(unconditioned)", "within each\nΔΔE stratum"],
                        fontsize=9)
    ax2.set_ylim(1.55, -.55)
    ax2.set_xlim(-.014, .125)
    ax2.set_xlabel("Δ AUC  (ours − Rosetta)", fontsize=9.5, color=SOFT)
    _style(ax2)

    fig.text(.070, .935, "Where does our advantage over Rosetta come from?",
             fontsize=14.5, color=INK, weight="semibold")
    fig.text(.070, .885,
             "Conditioning on conservation removes more than half the advantage, and what remains no longer "
             "clears zero. The first quantitative",
             fontsize=9.2, color=SOFT)
    fig.text(.070, .845,
             "signature that part of the Boltz ΔΔG is conservation — though the residual is positive in all "
             "four strata, so it is not ruled out.",
             fontsize=9.2, color=SOFT)
    fig.savefig(out, dpi=170)
    plt.close(fig)


VAMPSEQ = "005_NUDT15_abundance_reordered"
NUDT15_FN = "004_NUDT15_drug_sensitivity_reordered"


def vampseq_stats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The stability-readout contrast, and its within-protein bootstrap.

    NUDT15 contributes two of the 13 datasets — VAMP-seq abundance, which reads out
    cellular protein level and so is the closest thing in the corpus to a direct
    stability assay, and drug sensitivity, which reads out enzyme function. Same
    sequence, same structure, same alignment, and *the same ΔΔG predictions*: only
    the assay changes. That makes it a within-protein control on whether a predictor
    is measuring stability or merely ranking variants well in general.

    The bootstrap resamples the 156 POSITIONS, not the variants: inside one protein
    the position is the unit of independence, since a site's 19 substitutions share
    an environment, a burial and one alignment column. It answers "is this dataset's
    gap real", NOT "does it generalise to other proteins" — that is the 11-protein
    interval in `bootstrap.py`, and this one must not be quoted in its place.
    """
    def rho(a, b):
        from scipy.stats import spearmanr
        return abs(spearmanr(a, b).statistic)

    rows = []
    for ds, label in [(VAMPSEQ, "abundance (VAMP-seq)"),
                      (NUDT15_FN, "drug sensitivity")]:
        d = df[df["dataset"] == ds]
        rows.append(dict(dataset=ds, assay=label, n=len(d),
                         gemme=rho(d["dde"], d["s"]),
                         rosetta=rho(d["rosetta_ddg"], d["s"]),
                         boltz=rho(d["ddg_boltz"], d["s"])))
    table = pd.DataFrame(rows)

    v = df[df["dataset"] == VAMPSEQ].copy()
    v["pos"] = v["variant"].str.extract(r"(\d+)").astype(int)
    pos = v["pos"].unique()
    by = {p: v[v["pos"] == p] for p in pos}
    draws = {"rosetta": [], "gemme": []}
    for _ in range(4000):
        r = pd.concat([by[p] for p in RNG.choice(pos, len(pos), replace=True)])
        ours = rho(r["ddg_boltz"], r["s"])
        draws["rosetta"].append(ours - rho(r["rosetta_ddg"], r["s"]))
        draws["gemme"].append(ours - rho(r["dde"], r["s"]))
    boot = pd.DataFrame([
        dict(contrast=f"boltz_minus_{k}", n_variants=len(v), n_positions=len(pos),
             mean=float(np.mean(d)), ci_lo=float(np.percentile(d, 2.5)),
             ci_hi=float(np.percentile(d, 97.5)))
        for k, d in draws.items()])
    return table, boot


def fig_vampseq(table: pd.DataFrame, boot: pd.DataFrame, out: Path) -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.4, 4.6),
                                  gridspec_kw=dict(width_ratios=[1.5, 1], wspace=.30))
    fig.subplots_adjust(left=.070, right=.975, top=.755, bottom=.155)

    x = np.arange(2)
    w = .26
    for k, (col, name, c) in enumerate([("gemme", "GEMME ΔΔE (conservation)", GEM),
                                        ("rosetta", "Rosetta ΔΔG", ROS),
                                        ("boltz", "our ΔΔG", BOL)]):
        ax.bar(x + (k - 1) * w, table[col], w, color=c, label=name)
        for i, v in enumerate(table[col]):
            ax.text(i + (k - 1) * w, v + .012, f"{v:.3f}", ha="center",
                    fontsize=8.6, color=INK, weight="semibold")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.assay}\nn = {r.n:,}"
                        for r in table.itertuples()], fontsize=9.5)
    ax.set_ylim(0, .78)
    ax.set_ylabel("|Spearman ρ| against measured fitness", fontsize=9.5, color=SOFT)
    ax.set_xlabel("NUDT15 — same protein, same structure, same alignment, "
                  "the same predictions", fontsize=9, color=SOFT)
    ax.legend(frameon=False, fontsize=9, loc="upper right", ncol=1)
    _style(ax)

    ys = [0, 1]
    lbl = {"boltz_minus_rosetta": "ours − Rosetta", "boltz_minus_gemme": "ours − GEMME"}
    b = boot.set_index("contrast").loc[["boltz_minus_rosetta", "boltz_minus_gemme"]]
    ax2.barh(ys, b["mean"], .42, color=[BOL, "#8fb9ab"])
    ax2.errorbar(b["mean"], ys,
                 xerr=[b["mean"] - b["ci_lo"], b["ci_hi"] - b["mean"]],
                 fmt="none", ecolor=INK, lw=1.3, capsize=5)
    for y, (_, r) in zip(ys, b.iterrows()):
        ax2.text(r["ci_hi"] + .012, y, f"{r['mean']:+.3f}", va="center",
                 fontsize=10.5, color=INK, weight="semibold")
        ax2.text(r["ci_hi"] + .012, y + .28, f"[{r['ci_lo']:+.3f} · {r['ci_hi']:+.3f}]",
                 va="center", fontsize=8.4, color=SOFT)
    ax2.axvline(0, color=SOFT, lw=.9)
    ax2.set_yticks(ys)
    ax2.set_yticklabels([lbl[i] for i in b.index], fontsize=9.5)
    ax2.set_ylim(1.55, -.55)
    ax2.set_xlim(-.02, .62)
    ax2.set_xlabel("Δ|ρ| on the VAMP-seq dataset", fontsize=9.5, color=SOFT)
    _style(ax2)

    fig.text(.070, .935, "The stability assay inverts the ordering of the predictors",
             fontsize=14.5, color=INK, weight="semibold")
    fig.text(.070, .885,
             "On the function assay conservation wins; on the abundance assay — which reads out stability — "
             "both ΔΔG predictors win, and ours by more.",
             fontsize=9.2, color=SOFT)
    fig.text(.070, .845,
             "Right: 95 % CI from a cluster bootstrap over NUDT15's 156 positions. It answers whether this "
             "dataset's gap is real, not whether it generalises.",
             fontsize=9.2, color=SOFT)
    fig.savefig(out, dpi=170)
    plt.close(fig)


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--regime", default="mean",
                    help="which ΔΔG column of mave_ddg_predictions.csv to use")
    args = ap.parse_args(argv)

    FIGS.mkdir(parents=True, exist_ok=True)
    df = load(args.regime)
    sub = tertile_split(df)
    tol, dead = _check_orientation(sub)
    print(f"faithfulness check (Rosetta arm, our 13-dataset subset):")
    print(f"  low ΔΔE & low ΔΔG  -> {tol:.0f} % alta fitness   (paper: {PUBLISHED_CORNERS[0]:.0f} %)")
    print(f"  high ΔΔE & high ΔΔG -> {dead:.0f} % baja fitness  (paper: {PUBLISHED_CORNERS[1]:.0f} %)")
    print(f"  n = {len(df):,} variants, {len(sub):,} after dropping the middle tertile")

    cuts_b = matched_cuts(df)
    print(f"\nΔΔG sd: Rosetta {df.rosetta_ddg.std():.2f} · ours {df.ddg_boltz.std():.2f} kcal/mol")
    print(f"quantile-matched cuts for our arm: "
          f"{cuts_b[0]:.2f} / {cuts_b[1]:.2f} / {cuts_b[2]:.2f} kcal/mol")

    fig_landscape(df, sub, cuts_b, FIGS / "03_landscape_reproduction.png")
    print(f"\nwrote {FIGS / '03_landscape_reproduction.png'}")

    tab = strata_auc(sub)
    tab.to_csv(HERE / "conservation_strata_auc.csv", index=False)
    print("\nAUC for loss-of-function, within conservation strata:")
    print(tab[["stratum", "n", "pct_low", "auc_rosetta", "auc_boltz",
               "delta", "ci_lo", "ci_hi"]].to_string(index=False,
                                                     float_format=lambda v: f"{v:.3f}"))
    fig_strata(tab, FIGS / "04_conservation_strata.png")
    print(f"\nwrote {FIGS / '04_conservation_strata.png'}")
    print(f"wrote {HERE / 'conservation_strata_auc.csv'}")

    vtab, vboot = vampseq_stats(df)
    vtab.to_csv(HERE / "vampseq_dissociation.csv", index=False)
    vboot.to_csv(HERE / "vampseq_bootstrap.csv", index=False)
    print("\nNUDT15 — the same protein under two assays (|rho| vs measured fitness):")
    print(vtab[["assay", "n", "gemme", "rosetta", "boltz"]].to_string(
        index=False, float_format=lambda v: f"{v:.3f}"))
    print("\nVAMP-seq contrasts, cluster bootstrap over the 156 positions:")
    print(vboot.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    fig_vampseq(vtab, vboot, FIGS / "05_vampseq_dissociation.png")
    print(f"\nwrote {FIGS / '05_vampseq_dissociation.png'}")
    print(f"wrote {HERE / 'vampseq_dissociation.csv'}, {HERE / 'vampseq_bootstrap.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
