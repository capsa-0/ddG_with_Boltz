"""Paired head-to-head: this project's best transfer model vs. AFToolkit.

S669 is scored on the identical variants for both methods (this project's S669 corpus
is the 541 of 669 that survive the pipeline's 500-residue cap; AFToolkit's own
predictions for those same 541 come from `run_aftoolkit_s669.py`), so the comparison
is paired and the subset cannot flatter either side. Differences carry a paired
protein-cluster bootstrap.

FireProt is not a benchmark AFToolkit reports, and 47 of the 130 proteins in this
project's FireProt corpus are in AFToolkit's own training set, so the FireProt table
is reported on the subset that is blind to both methods.

    python results/16_aftoolkit_headtohead/headtohead.py
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
ANA = ROOT / "data/processed/_analysis"
B, SEED = 4000, 0
KEY = ["uniprot", "mutation", "ddg"]


def metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return dict(n=len(y), r=pearsonr(y, p)[0], rho=spearmanr(y, p).statistic,
                rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                mae=float(np.mean(np.abs(y - p))))


def paired_bootstrap(df, a, b, group="uniprot"):
    """Resample proteins with replacement; report the paired difference a - b."""
    rng = np.random.default_rng(SEED)
    prots = df[group].unique()
    idx = {p: df.index[df[group] == p].to_numpy() for p in prots}
    y, pa, pb = df.ddg_flip.to_numpy(), df[a].to_numpy(), df[b].to_numpy()
    dr, drho = [], []
    for _ in range(B):
        ii = np.concatenate([idx[p] for p in rng.choice(prots, len(prots), replace=True)])
        if y[ii].std() == 0:
            continue
        dr.append(pearsonr(y[ii], pa[ii])[0] - pearsonr(y[ii], pb[ii])[0])
        drho.append(spearmanr(y[ii], pa[ii]).statistic - spearmanr(y[ii], pb[ii]).statistic)
    q = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    return dict(d_r=float(np.mean(dr)), d_r_lo=q(dr)[0], d_r_hi=q(dr)[1],
                p_r_gt0=float(np.mean(np.array(dr) > 0)),
                d_rho=float(np.mean(drho)), d_rho_lo=q(drho)[0], d_rho_hi=q(drho)[1],
                p_rho_gt0=float(np.mean(np.array(drho) > 0)))


def s669():
    """Our per-variant S669 predictions (results/14 dumps) joined to AFToolkit's."""
    ours = None
    for f in ("exp14_s669_results_s669_locality.csv", "exp14_s669_results_onehot_s669.csv",
              "exp14_s669_results_s669_base.csv"):
        d = pd.read_csv(ANA / f).reset_index(drop=True)
        if ours is None:
            ours = d.copy()
        else:
            assert ours[["wt_id", "mutation", "ddg"]].equals(d[["wt_id", "mutation", "ddg"]])
            for c in d.columns:
                if c not in ours:
                    ours[c] = d[c].values
    # the dumps hold ddG already flipped into this project's sign convention
    ours = ours.rename(columns={"wt_id": "uniprot", "ddg": "ddg_flip"})
    ours["ddg"] = -ours.ddg_flip
    aft = pd.read_csv(R / "aftoolkit_s669_predictions.csv")
    # S669 contains 6 exactly-repeated (protein, mutation, ddG) triples; number them so
    # the join stays one-to-one instead of fanning out
    occ = lambda d: d.assign(_occ=d.groupby(KEY).cumcount())
    m = occ(ours).merge(occ(aft)[KEY + ["_occ", "aft_svm", "aft_mlp", "aft_catboost", "seqlen"]],
                        on=KEY + ["_occ"], how="left")
    assert len(m) == len(ours) and m.aft_svm.notna().all()
    return m


def main():
    m = s669()
    print(f"=== S669: {len(m)} variants / {m.uniprot.nunique()} proteins "
          f"(the subset both methods cover; AFToolkit's full 669 is reported separately)")
    named = [("aft_svm", "AFToolkit SVM (AF2 pair+lddt+plddt, 223,611 train)"),
             ("aft_mlp", "AFToolkit MLP"), ("aft_catboost", "AFToolkit CatBoost"),
             ("diag", "ours: Boltz-2 zdiag, 128d (best transfer)"),
             ("dz", "ours: Boltz-2 dz, 256d"),
             ("dz_cw", "ours: Boltz-2 diag + contact-weighted, 256d"),
             ("base", "ours: Boltz-2 concat, 256d (project default)")]
    rows = [dict(model=lab, **metrics(m.ddg_flip, m[c])) for c, lab in named if c in m]
    tab = pd.DataFrame(rows)
    print(tab.round(3).to_string(index=False))
    tab.to_csv(R / "headtohead_s669.csv", index=False)

    print("\npaired protein-cluster bootstrap on the same 541 variants "
          f"({B} resamples), ours - AFToolkit(SVM):")
    diffs = []
    for c, lab in [("diag", "zdiag 128d"), ("dz", "dz 256d"), ("base", "concat 256d")]:
        d = paired_bootstrap(m, c, "aft_svm")
        diffs.append(dict(ours=lab, vs="aft_svm", **d))
        print(f"  {lab:12s} dr={d['d_r']:+.3f} [{d['d_r_lo']:+.3f}, {d['d_r_hi']:+.3f}] "
              f"P={d['p_r_gt0']:.3f} | drho={d['d_rho']:+.3f} "
              f"[{d['d_rho_lo']:+.3f}, {d['d_rho_hi']:+.3f}] P={d['p_rho_gt0']:.3f}")
    pd.DataFrame(diffs).to_csv(R / "headtohead_s669_bootstrap.csv", index=False)

    # AFToolkit on the full benchmark, for the published-number check
    aft = pd.read_csv(R / "aftoolkit_s669_predictions.csv")
    y = aft.ddg.to_numpy(float)
    print(f"\nAFToolkit SVM on all 669 (published: rho 0.51, RMSE 1.41): "
          f"rho={spearmanr(y, -aft.aft_svm).statistic:.3f} "
          f"r={pearsonr(y, -aft.aft_svm)[0]:.3f} "
          f"rmse={np.sqrt(np.mean((y + aft.aft_svm) ** 2)):.3f}")
    kept = aft[aft.seqlen <= 500]
    print(f"AFToolkit SVM on the 128 variants our cap excludes: "
          f"rho={spearmanr(aft[aft.seqlen>500].ddg, -aft[aft.seqlen>500].aft_svm).statistic:.3f} "
          f"(on our {len(kept)}: {spearmanr(kept.ddg, -kept.aft_svm).statistic:.3f})")
    print(f"\nwrote {R/'headtohead_s669.csv'}, {R/'headtohead_s669_bootstrap.csv'}")


if __name__ == "__main__":
    main()
