"""
Where in the protein do Boltz and FoldX disagree?

The two methods live on wildly different scales (Boltz [-0.7, +4], FoldX [-2.7, +69]),
so a raw difference just re-plots FoldX's clash tail. Instead each method's ΔΔG is
converted to a **percentile rank** over the scanned mutations, and the discrepancy is

    delta = pct(Boltz) - pct(FoldX)          per mutation, in percentile points
    > 0  Boltz ranks it MORE destabilizing than FoldX does
    < 0  FoldX ranks it more destabilizing (its clash regime)

This is a DISAGREEMENT measure between two predictors, not an accuracy measure: there
is no measured ddG for this protein (see status.md), so neither method is arbitrated.
Two limits of the metric are worth keeping in view: the percentiles are relative to
THIS scanned set (31 of 38 positions are glycines, so it is not a neutral reference),
and rank-normalising both methods removes any disagreement about distribution by
construction -- including the large scale mismatch (regression slope Boltz~FoldX
= 0.08), which this metric cannot see.

Per position we report mean signed delta (direction of disagreement) and mean |delta|
(magnitude), then relate them to structure: burial, secondary structure, and distance
to the catalytic Asp pair (D170 nucleophile / D231 acid-base).

Outputs a B-factor-painted PDB so the map can be opened directly in PyMOL/ChimeraX.

    python results/10_gla_scan/map_discrepancy.py --pdb <1r46.pdb>
"""
import argparse
import math
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, DSSP
from Bio.PDB.Polypeptide import PPBuilder
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
CATALYTIC = (170, 231)          # human alpha-Gal A: D170 nucleophile, D231 acid/base
FLAGGED = [80, 137, 169, 200, 201, 228, 301, 325, 360, 409]


def structure_features(pdb_path, positions):
    """Per-residue burial, secondary structure and distance to the active site."""
    st = PDBParser(QUIET=True).get_structure("p", pdb_path)
    model = st[0]
    ch = model["A"]
    res = {r.id[1]: r for r in ch if r.id[0] == " "}

    # burial proxy: neighbouring residue centroids within 10 A
    cen = {i: (r["CB"] if "CB" in r else r["CA"]).coord
           for i, r in res.items() if ("CB" in r or "CA" in r)}
    ids = list(cen)
    crd = np.array([cen[i] for i in ids])
    d = np.linalg.norm(crd[:, None, :] - crd[None, :, :], axis=-1)
    nbr = dict(zip(ids, (d < 10).sum(1) - 1))

    # distance to the catalytic centre
    cat = [res[p]["CG"].coord for p in CATALYTIC if p in res and "CG" in res[p]]
    centre = np.mean(cat, axis=0) if cat else None
    dist = ({i: float(np.linalg.norm(cen[i] - centre)) for i in cen}
            if centre is not None else {})

    # secondary structure (DSSP), best-effort
    ss = {}
    try:
        for key, val in DSSP(model, str(pdb_path)).property_dict.items():
            if key[0] == "A":
                ss[key[1][1]] = val[2]
    except Exception as e:
        print(f"  (DSSP unavailable: {e})")

    return nbr, dist, ss, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", default=str(HERE / "compare_foldx_merged_mean.csv"))
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--out", default=str(HERE))
    args = ap.parse_args()
    out = Path(args.out)

    m = pd.read_csv(args.merged).dropna(subset=["boltz", "foldx"])
    # scale-free: percentile rank within the scanned set
    m["pct_boltz"] = m.boltz.rank(pct=True) * 100
    m["pct_foldx"] = m.foldx.rank(pct=True) * 100
    m["delta"] = m.pct_boltz - m.pct_foldx

    nbr, dist, ss, res = structure_features(args.pdb, sorted(m.position.unique()))

    per = m.groupby("position").agg(
        wt_aa=("wt_aa", "first"), n=("delta", "size"),
        delta_mean=("delta", "mean"), delta_absmean=("delta", lambda s: s.abs().mean()),
        boltz=("boltz", "mean"), foldx=("foldx", "mean")).reset_index()
    per["flagged"] = per.position.isin(FLAGGED)
    per["neighbors"] = per.position.map(nbr)
    per["dist_active"] = per.position.map(dist)
    per["ss"] = per.position.map(ss)
    per = per.sort_values("delta_mean")

    print("=== where the two methods disagree most (percentile points) ===")
    print("  delta = pct(Boltz) - pct(FoldX), each ranked within ITS OWN spread over")
    print("  these mutations. It measures DISAGREEMENT IN RELATIVE ORDERING between two")
    print("  predictors -- with no ground truth, it says nothing about which is correct.")
    print("  delta<0: FoldX ranks it more destabilizing")
    print("  delta>0: Boltz ranks it more destabilizing\n")
    print(f"  {'pos':>4} {'wt':>2} {'ss':>2} {'nbr':>4} {'dAct':>6} "
          f"{'delta':>7} {'|delta|':>8} {'Boltz':>7} {'FoldX':>7}")
    for r in per.itertuples(index=False):
        print(f"  {r.position:>4}{'*' if r.flagged else ' '}{r.wt_aa:>1} "
              f"{str(r.ss or '-'):>2} {r.neighbors:>4} {r.dist_active:>6.1f} "
              f"{r.delta_mean:>+7.1f} {r.delta_absmean:>8.1f} "
              f"{r.boltz:>+7.2f} {r.foldx:>+7.2f}")

    print("\n=== does the disagreement track structure? (Spearman over positions) ===")
    for col, lab in (("neighbors", "burial (neighbour count)"),
                     ("dist_active", "distance to catalytic centre")):
        ok = per[col].notna()
        rs = spearmanr(per[col][ok], per.delta_mean[ok]).statistic
        ra = spearmanr(per[col][ok], per.delta_absmean[ok]).statistic
        print(f"  {lab:<32} vs signed delta {rs:+.3f} | vs |delta| {ra:+.3f}")
    if per.ss.notna().any():
        print("\n  by secondary structure:")
        for s, g in per.groupby(per.ss.fillna("-")):
            print(f"    {s:>2}  n={len(g):>2}  mean delta {g.delta_mean.mean():+6.1f}  "
                  f"mean |delta| {g.delta_absmean.mean():5.1f}")

    # raw-scale difference, in kcal/mol, for comparison with the rank version
    raw = m.assign(diff=m.boltz - m.foldx).groupby("position").agg(
        diff_mean=("diff", "mean")).reset_index()
    per = per.merge(raw, on="position", how="left")
    _raw_diagnostics(per)

    per.to_csv(out / "discrepancy_by_position.csv", index=False)
    _paint(args.pdb, per, out / "boltz_minus_foldx.pdb")
    _figure(m, per, out / "figures" / "02_discrepancy_map.png")
    _figure_raw(m, per, out / "figures" / "03_discrepancy_map_raw.png")
    print(f"\nwrote -> {out}")


def _raw_diagnostics(per):
    """How much of the raw difference is just -FoldX?"""
    print("\n=== raw-scale difference (kcal/mol) vs each method ===")
    for col, lab in (("foldx", "FoldX per-position mean"),
                     ("boltz", "Boltz per-position mean")):
        r = spearmanr(per[col], per.diff_mean).statistic
        print(f"  corr(Boltz-FoldX difference, {lab:<26}) = {r:+.3f}")
    sd_b, sd_f = per.boltz.std(), per.foldx.std()
    print(f"  per-position SD: Boltz {sd_b:.2f} vs FoldX {sd_f:.2f} kcal/mol "
          f"(FoldX varies {sd_f/sd_b:.1f}x more)")
    print(f"  raw difference range: [{per.diff_mean.min():+.2f}, "
          f"{per.diff_mean.max():+.2f}] kcal/mol")


def _figure_raw(m, per, path):
    """Same map on the REAL kcal/mol scale, with a broken axis for FoldX's tail."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    path.parent.mkdir(exist_ok=True)
    o = per.sort_values("position").reset_index(drop=True)
    x = np.arange(len(o))
    is_g = (o.wt_aa == "G").to_numpy()
    fig, (a, b) = plt.subplots(2, 1, figsize=(14, 8),
                               gridspec_kw=dict(height_ratios=[2, 2]), sharex=True)

    for i, r in enumerate(o.itertuples(index=False)):
        if r.flagged:
            for ax in (a, b):
                ax.axvspan(i - .5, i + .5, color="#DDAA33", alpha=.22, zorder=0)

    a.bar(x, o.diff_mean, color="#C44E52",
          edgecolor=["black" if g else "none" for g in is_g],
          linewidth=[1.1 if g else 0 for g in is_g], zorder=2)
    a.axhline(0, color="0.3", lw=.8)
    a.set_ylabel("Boltz − FoldX\n(kcal/mol, REAL values)")
    a.set_title("Same map on the real kcal/mol scale — the difference is essentially "
                "−FoldX\n(Boltz's whole range is thinner than one FoldX bar)", fontsize=10)
    a.legend(handles=[
        Patch(facecolor="white", edgecolor="black", linewidth=1.1, label="wild-type glycine"),
        Patch(facecolor="#DDAA33", alpha=.5, label="flagged position"),
    ], fontsize=8, loc="lower left", framealpha=.92)

    w = 0.4
    b.bar(x - w/2, o.boltz, width=w, color="#4C72B0", label="Boltz", zorder=2)
    b.bar(x + w/2, o.foldx, width=w, color="#DD8452", label="FoldX", zorder=2)
    b.axhline(0, color="0.3", lw=.8)
    b.set_ylabel("mean ΔΔG at that position\n(kcal/mol)")
    b.set_xticks(x)
    b.set_xticklabels([f"{r.wt_aa}{r.position}" + ("*" if r.flagged else "")
                       for r in o.itertuples(index=False)], rotation=90, fontsize=6)
    for tick, fl in zip(b.get_xticklabels(), o.flagged):
        if fl:
            tick.set_color("#B8860B"); tick.set_fontweight("bold")
    b.legend(fontsize=8); b.grid(alpha=.3)
    b.set_xlim(-0.8, len(o) - 0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _paint(pdb_path, per, out_pdb):
    """B-factor = signed delta; occupancy = 1 for scanned positions, 0 otherwise."""
    st = PDBParser(QUIET=True).get_structure("p", pdb_path)
    val = dict(zip(per.position, per.delta_mean))
    for atom in st.get_atoms():
        rid = atom.get_parent().id[1]
        if rid in val:
            atom.set_bfactor(round(float(val[rid]), 2))
            atom.set_occupancy(1.00)
        else:
            atom.set_bfactor(0.00)
            atom.set_occupancy(0.00)
    io = PDBIO(); io.set_structure(st); io.save(str(out_pdb))
    print(f"\npainted structure -> {out_pdb}")
    print("  PyMOL:  load boltz_minus_foldx.pdb; select scanned, occupancy > 0.5")
    print("          spectrum b, blue_white_red, scanned, -60, 60; show spheres, scanned")


def _figure(m, per, path):
    """Discrepancy along the sequence, with flagged positions and glycines marked.

    Two orthogonal annotations, because they overlap (3 of the 10 flagged positions
    are themselves glycines): flagged = shaded band + bold red tick label; glycine =
    black-outlined bar + 'G' marker strip under the axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    path.parent.mkdir(exist_ok=True)
    o = per.sort_values("position").reset_index(drop=True)
    x = np.arange(len(o))
    is_g = (o.wt_aa == "G").to_numpy()
    fig, (a, b, c) = plt.subplots(3, 1, figsize=(14, 9.5), sharex=True,
                                  gridspec_kw=dict(height_ratios=[2, 2, 1.3]))

    for i, r in enumerate(o.itertuples(index=False)):
        if r.flagged:
            for ax in (a, b, c):
                ax.axvspan(i - .5, i + .5, color="#DDAA33", alpha=.22, zorder=0)

    a.bar(x, o.delta_mean,
          color=["#C44E52" if v < 0 else "#4C72B0" for v in o.delta_mean],
          edgecolor=["black" if g else "none" for g in is_g],
          linewidth=[1.1 if g else 0 for g in is_g], zorder=2)
    a.axhline(0, color="0.3", lw=0.8)
    a.set_ylabel("Boltz − FoldX\n(percentile points)")
    a.set_title("Disagreement in relative ordering (no ground truth: neither is 'right')\n"
                "blue = Boltz ranks it more destabilizing · red = FoldX does", fontsize=10)
    a.legend(handles=[
        Patch(facecolor="#4C72B0", label="Boltz ranks more destabilizing"),
        Patch(facecolor="#C44E52", label="FoldX ranks more destabilizing"),
        Patch(facecolor="white", edgecolor="black", linewidth=1.1,
              label="wild-type glycine (31 of 38)"),
        Patch(facecolor="#DDAA33", alpha=.5, label="flagged position (10)"),
    ], fontsize=8, ncol=2, loc="upper left", framealpha=.92)

    b.plot(x, o.boltz.rank(pct=True) * 100, "o-", ms=4, lw=1.2, color="#4C72B0",
           label="Boltz (percentile)", zorder=2)
    b.plot(x, o.foldx.rank(pct=True) * 100, "s-", ms=4, lw=1.2, color="#DD8452",
           label="FoldX (percentile)", zorder=2)
    b.set_ylabel("per-position severity\n(percentile)")
    b.legend(fontsize=8); b.grid(alpha=.3)

    c.bar(x, o.neighbors, color="#55A868",
          edgecolor=["black" if g else "none" for g in is_g],
          linewidth=[1.1 if g else 0 for g in is_g], zorder=2)
    c.set_ylabel("burial\n(neighbours)")
    # 'G' strip marking every glycine position
    for i, g in enumerate(is_g):
        if g:
            c.annotate("G", (i, 0), xytext=(0, -22), textcoords="offset points",
                       ha="center", va="top", fontsize=6.5, color="#333333",
                       annotation_clip=False, fontweight="bold")
    c.set_xticks(x)
    c.set_xticklabels([f"{r.wt_aa}{r.position}" + ("*" if r.flagged else "")
                       for r in o.itertuples(index=False)], rotation=90, fontsize=6)
    for tick, fl in zip(c.get_xticklabels(), o.flagged):
        if fl:
            tick.set_color("#B8860B"); tick.set_fontweight("bold")
    c.grid(alpha=.3)
    c.set_xlim(-0.8, len(o) - 0.2)
    fig.text(0.5, 0.005, "* = flagged position (shaded) · outlined bars and the G strip "
             "= wild-type glycine", ha="center", fontsize=8, color="0.3")
    fig.tight_layout(rect=[0, 0.022, 1, 1])
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
