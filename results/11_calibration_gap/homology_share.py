"""Do similar proteins carry similar dG(WT) / similar offsets?

If yes, the quantity is a function of the fold and is predictable in principle
(just not by what we tried). If homologues have uncorrelated values, it is not a
property of the fold at all and no sequence/structure model can recover it.

Levels of similarity, tightest first:
  same base PDB   e.g. 1A0N.pdb_L7S vs 1A0N.pdb_V55A -- same protein, different background
  cluster 90/50/30 %  MMseqs2 clusters already computed for this corpus
  random pairs        the null
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
SCR = ROOT / "data/processed/_analysis"
HERE = Path(__file__).resolve().parent
SCR.mkdir(parents=True, exist_ok=True)


ROWS = []
CURRENT = [""]


def r(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if min(a.std(), b.std()) > 1e-9 else np.nan


def paired_stats(vals, labels, name, n_rep=200, seed=0):
    """Sample one random within-group pair per group, repeatedly; report r and |diff|."""
    s = pd.DataFrame({"v": vals, "g": labels}).dropna()
    sizes = s.groupby("g").size()
    usable = sizes[sizes >= 2].index
    if len(usable) < 8:
        print(f"  {name:22s} — only {len(usable)} groups with >=2 members, skipped")
        return
    rng = np.random.default_rng(seed)
    rs, ds = [], []
    for _ in range(n_rep):
        a, b = [], []
        for g in usable:
            v = s.loc[s.g == g, "v"].to_numpy()
            i, j = rng.choice(len(v), 2, replace=False)
            a.append(v[i]); b.append(v[j])
        rs.append(r(a, b))
        ds.append(np.mean(np.abs(np.array(a) - np.array(b))))
    # null: random pairs across the whole set
    nr, nd = [], []
    v = s.v.to_numpy()
    for _ in range(n_rep):
        i = rng.permutation(len(v)); j = rng.permutation(len(v))
        k = min(len(usable), len(v) // 2)
        nr.append(r(v[i[:k]], v[j[:k]]))
        nd.append(np.mean(np.abs(v[i[:k]] - v[j[:k]])))
    gm = s.groupby("g").v.mean()
    vb = gm.var()
    vw = s.groupby("g").v.transform("mean").rsub(s.v).var()
    icc = vb / (vb + vw) if (vb + vw) > 0 else np.nan
    print(f"  {name:22s} n_groups={len(usable):3d}  pair r={np.mean(rs):+.3f}±{np.std(rs):.3f}  "
          f"|Δ|={np.mean(ds):.2f} (random {np.mean(nd):.2f})  ICC={icc:.3f}")
    # The decisive comparison of this experiment lives in these rows -- homologues
    # share the per-protein MEAN ddG but not the model's ERROR on it. Persist them.
    ROWS.append(dict(quantity=CURRENT[0], grouping=name.strip(), n_groups=len(usable),
                     pair_r=float(np.mean(rs)), pair_r_sd=float(np.std(rs)),
                     mean_abs_diff=float(np.mean(ds)), random_abs_diff=float(np.mean(nd)),
                     icc=float(icc)))


def main():
    dg = pd.read_csv(SCR / "tsu_dG_wt.csv")           # wt_id, dG_wt, cluster(30%)
    ids = dg.wt_id.tolist()
    dg["base"] = dg.wt_id.str.split("_").str[0]       # 1A0N.pdb_L7S -> 1A0N.pdb

    cl = {}
    for thr in (30, 50, 90):
        p = ROOT / f"data/processed/tsuboyama_bench_fast/cluster_map_{thr}.csv"
        cl[thr] = pd.read_csv(p).set_index("protein_id").cluster
        dg[f"c{thr}"] = dg.wt_id.map(cl[thr])

    print(f"n proteins {len(dg)}; distinct base structures {dg.base.nunique()}; "
          f"clusters 90/50/30 = {dg.c90.nunique()}/{dg.c50.nunique()}/{dg.c30.nunique()}")
    print(f"dG(WT): sd {dg.dG_wt.std():.2f} kcal/mol\n")

    CURRENT[0] = "dG_wt"
    print("=== dG(WT): do similar proteins share it? ===")
    for lab, col in (("same base structure", "base"), ("same cluster 90%", "c90"),
                     ("same cluster 50%", "c50"), ("same cluster 30%", "c30")):
        paired_stats(dg.dG_wt.to_numpy(), dg[col].to_numpy(), lab)

    # ---- same question for the per-protein offset, if OOF preds exist ----
    f = SCR / "tsu_mut_classes.csv"
    if not f.exists():
        print("\n[offset: OOF predictions not written yet — rerun after exp1]")
        return
    d = pd.read_csv(f)
    off = (d.y - d.pred_OOF).groupby(d.wt_id).mean()
    mean_ddg = d.groupby("wt_id").y.mean()
    n_per = d.groupby("wt_id").size()
    keep = n_per[n_per >= 6].index

    t = pd.DataFrame({"wt_id": off.index, "off": off.to_numpy(),
                      "mean_ddg": mean_ddg.loc[off.index].to_numpy()})
    t = t[t.wt_id.isin(keep)]
    t["base"] = t.wt_id.str.split("_").str[0]
    for thr in (30, 50, 90):
        t[f"c{thr}"] = t.wt_id.map(cl[thr])

    CURRENT[0] = "offset"
    print(f"\n=== per-protein OFFSET (n={len(t)} proteins with >=6 muts, "
          f"sd {t.off.std():.2f}) ===")
    for lab, col in (("same base structure", "base"), ("same cluster 90%", "c90"),
                     ("same cluster 50%", "c50"), ("same cluster 30%", "c30")):
        paired_stats(t.off.to_numpy(), t[col].to_numpy(), lab)

    CURRENT[0] = "mean_ddg"
    print(f"\n=== per-protein MEAN ddG (the quantity itself, sd {t.mean_ddg.std():.2f}) ===")
    for lab, col in (("same base structure", "base"), ("same cluster 90%", "c90"),
                     ("same cluster 50%", "c50"), ("same cluster 30%", "c30")):
        paired_stats(t.mean_ddg.to_numpy(), t[col].to_numpy(), lab)

    m = t.merge(dg[["wt_id", "dG_wt"]], on="wt_id", how="inner")
    print(f"\n[reference] dG(WT) vs per-protein mean ddG: r={r(m.dG_wt, m.mean_ddg):+.3f} "
          f"| dG(WT) vs offset: r={r(m.dG_wt, m.off):+.3f}  (n={len(m)})")


if __name__ == "__main__":
    main()

    pd.DataFrame(ROWS).to_csv(HERE / "homology_share.csv", index=False)
    print(f"\nwrote {HERE / 'homology_share.csv'}")
