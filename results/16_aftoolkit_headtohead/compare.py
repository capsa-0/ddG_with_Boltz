"""Head-to-head: this project's best transfer model vs. AFToolkit, on S669 and FireProt.

    python results/16_aftoolkit_headtohead/compare.py

Reads the per-variant prediction dumps written by results/14 (`--transfer s669` /
`--transfer fireprot_le500`, Tsuboyama-trained, blind), re-derives the homology
filters from the committed cluster maps, scores every configuration on the metric
set AFToolkit reports, attaches a protein-cluster bootstrap CI, and measures how much
of the FireProt corpus is present in AFToolkit's own training pool.

Writes: results_ours.csv, bootstrap_ci.csv, fireprot_aftoolkit_train_overlap.csv.
"""
import os

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import roc_auc_score, matthews_corrcoef

R = Path(__file__).resolve().parent
ROOT = R.parents[1]
ANA = ROOT / "data/processed/_analysis"
KEY = ["wt_id", "mutation", "ddg"]
B = 4000                      # bootstrap resamples
SEED = 0

# The dumps store ddG already in this project's convention (S669's sign is flipped at
# load time by run_ablation.py), so negative == stabilizing everywhere below.
S669_DUMPS = ["exp14_s669_results_s669_locality.csv", "exp14_s669_results_onehot_s669.csv",
              "exp14_s669_results_s669_base.csv"]
# all no-augmentation dumps, so every configuration is compared under one protocol
FP_DUMPS = ["exp14_fpfilt_results_locality_paired.csv", "exp14_fpfilt_results_onehot_fp.csv",
            "exp14_fpfilt_results_farctrl.csv", "exp14_fpfilt_results_fact_noaug.csv"]
CONFIGS = ["diag", "dz_cw", "dz", "cw", "base", "far", "onehot"]
AFT = Path(os.environ.get("AFT", "/tmp/aftoolkit"))   # AFToolkit assets, see run_aftoolkit_s669.py


def merge_positional(files):
    """The dumps share one row order; S669 has 17 repeated (wt_id, mutation) pairs, so
    merging on the key would fan out. Concatenate by position and assert alignment."""
    base = None
    for f in files:
        d = pd.read_csv(ANA / f).reset_index(drop=True)
        if base is None:
            base = d.copy()
            continue
        assert base[KEY].equals(d[KEY]), f"row order differs in {f}"
        for c in d.columns:
            if c not in base.columns:
                base[c] = d[c].values
    return base


def metrics(y, p, groups):
    y, p = np.asarray(y, float), np.asarray(p, float)
    stab = y < 0
    per = (pd.DataFrame(dict(g=list(groups), y=y, p=p)).groupby("g")
           .apply(lambda d: pearsonr(d.y, d.p)[0]
                  if len(d) > 2 and d.y.std() > 0 and d.p.std() > 0 else np.nan,
                  include_groups=False))
    return dict(n=len(y), n_prot=len(set(groups)),
                r=pearsonr(y, p)[0], rho=spearmanr(y, p)[0],
                rmse=float(np.sqrt(np.mean((y - p) ** 2))),
                mae=float(np.mean(np.abs(y - p))),
                auc_stab=roc_auc_score(stab, -p), mcc_stab=matthews_corrcoef(stab, p < 0),
                per_prot_median_r=float(np.nanmedian(per)), n_prot_scored=int(per.notna().sum()))


def bootstrap(df, col, b=B):
    """Resample whole proteins with replacement (variants within a protein are not
    independent), then take percentile CIs for r and rho."""
    rng = np.random.default_rng(SEED)
    prots = df.wt_id.unique()
    idx = {p: df.index[df.wt_id == p].to_numpy() for p in prots}
    y_all, p_all = df.ddg.to_numpy(), df[col].to_numpy()
    rs, rhos = [], []
    for _ in range(b):
        ii = np.concatenate([idx[p] for p in rng.choice(prots, len(prots), replace=True)])
        y, p = y_all[ii], p_all[ii]
        if y.std() == 0 or p.std() == 0:
            continue
        rs.append(pearsonr(y, p)[0])
        rhos.append(spearmanr(y, p)[0])
    q = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return q(rs), q(rhos)


# ---------------------------------------------------------------- homology re-check
def homology_report(fp_dump):
    """Re-derive both filters from the committed cluster maps rather than trusting the
    protein lists baked into the dumps."""
    leak = pd.read_csv(ROOT / "results/09_external_benchmarks/homology/s669_leakage.csv")
    print(f"S669 leakage vs the training corpus (MMseqs2, 80% coverage), "
          f"{len(leak)} proteins / {int(leak.n_variants.sum())} variants:")
    for thr in (25, 30):
        print(f"  vs Tsuboyama @{thr}% identity : {int(leak[f'leaky_tsu_{thr}'].sum())} leaky proteins")
        print(f"  vs FireProt  @{thr}% identity : {int(leak[f'leaky_fp_{thr}'].sum())} leaky proteins")

    cmap = pd.read_csv(ROOT / "results/08_finetune_fireprot/splits/cluster_map_30.csv")
    # Tsuboyama wt_ids already carry the ".pdb" suffix the map uses; FireProt's do not.
    cmap["wt_id"] = cmap.protein_id
    cmap["fp_id"] = cmap.protein_id.str.replace(".pdb", "", regex=False)
    tsu = set(pd.read_csv(ROOT / "data/processed/tsuboyama_bench_fast/mutations.csv").wt_id)
    fp_all = pd.concat([pd.read_csv(ROOT / f"data/processed/{d}/mutations.csv")
                        for d in ("fireprot_le200", "fireprot_201to500")], ignore_index=True)
    tsu_clusters = set(cmap[cmap.wt_id.isin(tsu)].cluster)
    assert len(tsu_clusters), "no Tsuboyama proteins matched the cluster map"
    leaky_fp = set(cmap[cmap.fp_id.isin(set(fp_all.wt_id))
                        & cmap.cluster.isin(tsu_clusters)].fp_id)
    kept = set(fp_dump.wt_id)
    print(f"\nFireProt <=500 vs Tsuboyama @30% identity: {len(leaky_fp)} of "
          f"{fp_all.wt_id.nunique()} proteins leaky "
          f"({(fp_all.wt_id.isin(leaky_fp)).sum()} of {len(fp_all)} variants)")
    print(f"  scored corpus = {len(kept)} proteins / {len(fp_dump)} variants; "
          f"leaky proteins present in it: {len(kept & leaky_fp)} (must be 0)")
    assert not (kept & leaky_fp)
    return fp_all, leaky_fp


# ------------------------------------------------- AFToolkit's own training overlap
def aftoolkit_overlap(fp_dump, fp_all):
    """AFToolkit trains on cDNA (=Megascale) + 2,375 PROSTATA rows, and excludes proteins
    homologous to ITS OWN test sets (S669/Ssym/PTMul; BLAST >36% identity, e<0.05).
    Nothing in that procedure removes FireProt, so measure the overlap directly against
    AFToolkit's released training manifest (`cdna+PROSTATA_mut_idxs.csv`, shipped in
    stability_task_files.zip)."""
    man = AFT / "cdna+PROSTATA_mut_idxs.csv"
    if not man.exists():
        print(f"\n[skip] AFToolkit training manifest not found at {man}; "
              f"see run_aftoolkit_s669.py for the download")
        return None
    aft = pd.read_csv(man, index_col=0)
    aft = aft[aft.split == "train"]
    train_pdbs = {p.upper() for p in aft.pdb_id.unique()}
    raw = pd.concat([pd.read_csv(ROOT / f"data/raw/fireprot_{d}.csv")
                     for d in ("le200", "201to500")], ignore_index=True)
    raw["wt_id"] = raw.uniprot_id.fillna(raw.pdb_id)
    raw["pdb1"] = raw.pdb_id.astype(str).str.split("|").str[0].str.upper()
    raw["mut_info"] = raw.wild_type + raw.position.astype(str) + raw.mutation
    pdbmap = raw.groupby("wt_id").pdb1.first()

    fp = fp_dump[["wt_id", "mutation", "ddg"]].copy()
    fp["pdb"] = fp.wt_id.map(pdbmap)
    fp["protein_in_aft_train"] = fp.pdb.isin(train_pdbs)
    exact = {(r.pdb_id.upper(), r.mut_info) for r in aft[aft.source == "prostata"].itertuples()}
    cand = raw[["wt_id", "pdb1", "mut_info"]].drop_duplicates()
    cand["variant_in_aft_train"] = [(r.pdb1, r.mut_info) in exact for r in cand.itertuples()]
    fp = fp.merge(cand.rename(columns={"mut_info": "mutation"})[
        ["wt_id", "mutation", "variant_in_aft_train"]].drop_duplicates(),
        on=["wt_id", "mutation"], how="left")
    fp["variant_in_aft_train"] = fp.variant_in_aft_train.fillna(False)

    print(f"\nAFToolkit training set: {len(aft)} rows "
          f"({(aft.source=='mega').sum()} cDNA/Megascale + {(aft.source=='prostata').sum()} "
          f"PROSTATA) on {aft.pdb_id.nunique()} proteins")
    print(f"  FireProt proteins that ARE in it: "
          f"{fp.loc[fp.protein_in_aft_train,'wt_id'].nunique()} of {fp.wt_id.nunique()} "
          f"({fp.protein_in_aft_train.sum()} of {len(fp)} variants, "
          f"{fp.protein_in_aft_train.mean():.1%})")
    print(f"  identical (protein, mutation) pairs: {int(fp.variant_in_aft_train.sum())}")
    clean = fp[~fp.protein_in_aft_train]
    print(f"  -> subset blind to BOTH methods: {len(clean)} variants / "
          f"{clean.wt_id.nunique()} proteins")
    fp.to_csv(R / "fireprot_aftoolkit_train_overlap.csv", index=False)
    return fp


def main():
    s669 = merge_positional(S669_DUMPS).reset_index(drop=True)
    fpf = merge_positional(FP_DUMPS).reset_index(drop=True)
    fp_all, _ = homology_report(fpf)

    rows, cis = [], []
    for tag, df in (("s669", s669), ("fireprot_le500_filtered", fpf)):
        print(f"\n=== {tag}: {len(df)} variants / {df.wt_id.nunique()} proteins, "
              f"{(df.ddg < 0).mean():.1%} stabilizing")
        for c in CONFIGS:
            if c not in df:
                continue
            rows.append(dict(corpus=tag, config=c, **metrics(df.ddg, df[c], df.wt_id)))
            (rl, rh), (sl, sh) = bootstrap(df, c)
            cis.append(dict(corpus=tag, config=c, r=rows[-1]["r"], r_lo=rl, r_hi=rh,
                            rho=rows[-1]["rho"], rho_lo=sl, rho_hi=sh))
        print(pd.DataFrame([r for r in rows if r["corpus"] == tag]).round(3).to_string(index=False))

    res = pd.DataFrame(rows)
    res.to_csv(R / "results_ours.csv", index=False)
    ci = pd.DataFrame(cis)
    ci.to_csv(R / "bootstrap_ci.csv", index=False)
    print(f"\nprotein-cluster bootstrap ({B} resamples):")
    print(ci.round(3).to_string(index=False))

    ov = aftoolkit_overlap(fpf, fp_all)
    if ov is not None:
        clean = fpf.merge(ov.loc[~ov.protein_in_aft_train, ["wt_id", "mutation"]]
                          .drop_duplicates(), on=["wt_id", "mutation"])
        print(f"\n=== FireProt subset blind to both methods: {len(clean)} variants / "
              f"{clean.wt_id.nunique()} proteins")
        rows = [dict(corpus="fireprot_blind_to_both", config=c,
                     **metrics(clean.ddg, clean[c], clean.wt_id))
                for c in CONFIGS if c in clean]
        print(pd.DataFrame(rows).round(3).to_string(index=False))
        pd.concat([res, pd.DataFrame(rows)]).to_csv(R / "results_ours.csv", index=False)
    print(f"\nwrote {R/'results_ours.csv'}, {R/'bootstrap_ci.csv'}, "
          f"{R/'fireprot_aftoolkit_train_overlap.csv'}")


if __name__ == "__main__":
    main()
