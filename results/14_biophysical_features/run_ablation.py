"""14 — Feature-block ablation: does biology-informed featurisation help?

Protocol fixed to results/06 / 07 / 13 so the numbers join that series:
  * corpus   tsuboyama_bench_fast (12,359 muts / 412 proteins)
  * split    GroupKFold(5) on wt_id, out-of-fold predictions
  * model    make_model("mlp") -- impute -> scale -> 5-seed MLP(256,128,64) ensemble
  * augment  antisymmetry on TRAIN folds only (results/07 default)
  * transfer the same fold models are averaged onto S669 (541 variants, sign flipped)

Every block is declared as (invariant, wt-side, mt-side) columns so the antisymmetry
augmentation stays exact: reversing a mutation swaps the wt and mt halves and negates
ddG, and leaves site properties (burial, chain length, position) untouched.

Reported per configuration: overall r / rho / MAE **and** the stabilizing-tail metrics
from results/13 (STAB = -0.5). Pooled r is data-saturated (results/03), so the tail
metrics are the primary endpoint -- see notes/plans/00_overview.md.

    python results/14_biophysical_features/run_ablation.py [--configs a b c]
"""
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from ddg.evaluation.models import make_model

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
OUT = ROOT / "results/14_biophysical_features"
Z, K = 128, 5
STAB = -0.5          # ddG < -0.5 kcal/mol == experimentally stabilizing (results/13)
N_JOBS = 2           # 4 cores / ~2 GB free on this box; does not change the fit

RES_KEYS = ("vol", "hyd", "dgtrans", "charge", "polar", "aromatic",
            "helixprop", "sheetprop", "flex", "is_gly", "is_pro")
INTER_KEYS = ("x_vol", "x_hyd", "x_dgtrans", "x_charge", "x_is_gly", "x_is_pro")
SITE_COLS = ["site_cn8", "site_cn10", "site_cn12", "site_cn_z",
             "site_relpos", "site_len", "site_termdist"]
# The size-dependent site scalars do not transfer: Tsuboyama chains are 32-72 aa
# (site_len mean 53), S669's are 50-493 (mean 281), so a Tsuboyama-trained model is
# extrapolating far outside its observed range. site_cn_z (burial z-scored WITHIN the
# protein) and site_relpos are dimensionless and matched across corpora by
# construction; this is the transferable subset.
SITE_COLS_T = ["site_cn_z", "site_relpos"]

# item 3 — MSA conservation. msa_has_msa / msa_neff are what let the model discount
# the block on de novo designed proteins, which have no natural homologues at all.
MSA_SITE_COLS = ["msa_neff", "msa_depth", "msa_gapfrac", "msa_entropy",
                 "msa_maxfreq", "msa_has_msa"]
MSA_RES_KEYS = ("msafreq", "msalogodds", "is_consensus", "x_cons")


def zblock(wt_prefix, mt_prefix):
    return ([], [f"{wt_prefix}_{j}" for j in range(Z)],
            [f"{mt_prefix}_{j}" for j in range(Z)])


# name -> (invariant cols, wt-side cols, mt-side cols)
BLOCKS = {
    "z":       zblock("wtz", "mtz"),          # the current pipeline's concat block
    "cw":      zblock("wtcw", "mtcw"),        # item 1: contact-weighted pooling
    "far":     zblock("wtfar", "mtfar"),      # item 1: the far-shell control
    "bio":     (SITE_COLS,
                [f"wt_{k}" for k in RES_KEYS] + [f"wt_{k}" for k in INTER_KEYS],
                [f"mt_{k}" for k in RES_KEYS] + [f"mt_{k}" for k in INTER_KEYS]),
    "bio_nox": (SITE_COLS,                    # item 2 without the interaction terms
                [f"wt_{k}" for k in RES_KEYS],
                [f"mt_{k}" for k in RES_KEYS]),
    "bio_t":   (SITE_COLS_T,                  # item 2 with only transferable site cols
                [f"wt_{k}" for k in RES_KEYS] + [f"wt_{k}" for k in INTER_KEYS],
                [f"mt_{k}" for k in RES_KEYS] + [f"mt_{k}" for k in INTER_KEYS]),
    "cons":    (MSA_SITE_COLS,                # item 3: conservation / PSSM / consensus
                [f"wt_{k}" for k in MSA_RES_KEYS],
                [f"mt_{k}" for k in MSA_RES_KEYS]),
}

CONFIGS = {
    "base":          ["z"],                   # sanity gate: must reproduce r ~ 0.799
    "cw":            ["cw"],                  # does contact weighting REPLACE uniform?
    "base+cw":       ["z", "cw"],             # is it complementary?
    "cw+far":        ["cw", "far"],           # capacity control for base+cw (512 dims)
    "bio":           ["bio"],                 # 40 hand-made numbers, alone
    "base+bio":      ["z", "bio"],
    "base+bio_nox":  ["z", "bio_nox"],        # is the gain the interactions?
    "base+cw+bio":   ["z", "cw", "bio"],
    "cw+bio_t":      ["cw", "bio_t"],         # does bio still hurt once size is out?
    "base+cw+bio_t": ["z", "cw", "bio_t"],
    # item 3 (needs features_msa.parquet)
    "cons":          ["cons"],                # how far does conservation get alone?
    "base+cons":     ["z", "cons"],           # does it add to Boltz's implicit MSA use?
    "cw+cons":       ["cw", "cons"],
    "all":           ["z", "cw", "bio_t", "cons"],
}


def assemble(names):
    """Concatenate blocks into (columns, n_invariant, n_side)."""
    inv, wt, mt = [], [], []
    for n in names:
        i, w, m = BLOCKS[n]
        inv += i
        wt += w
        mt += m
    assert len(wt) == len(mt), "wt/mt sides must pair 1:1 for the swap"
    return inv + wt + mt, len(inv), len(wt)


def augment(X, y, n_inv, n_side):
    """Antisymmetry: swap the wt/mt halves, negate ddG. Site block is untouched."""
    inv = X[:, :n_inv]
    wt = X[:, n_inv:n_inv + n_side]
    mt = X[:, n_inv + n_side:]
    Xs = np.concatenate([inv, mt, wt], axis=1)
    return np.vstack([X, Xs]), np.concatenate([y, -y])


def ndcg_stab(y, pred, k=30):
    """nDCG over the stabilizing ranking: gain = max(0, -ddG). (results/13)"""
    gain = np.maximum(0.0, -y)
    order = np.argsort(pred)                 # most-stabilizing prediction first
    disc = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = float((gain[order[:k]] * disc).sum())
    ideal = float((np.sort(gain)[::-1][:k] * disc).sum())
    return dcg / ideal if ideal > 0 else np.nan


def metrics(y, pred, tag, cfg):
    stab = y < STAB
    return {
        "config": cfg, "set": tag, "n": int(len(y)), "n_stab": int(stab.sum()),
        "r": float(np.corrcoef(y, pred)[0, 1]),
        "rho": float(spearmanr(y, pred).statistic),
        "mae": float(np.abs(pred - y).mean()),
        "rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        "stab_bias": float((pred[stab] - y[stab]).mean()) if stab.any() else np.nan,
        "stab_rho": float(spearmanr(y[stab], pred[stab]).statistic) if stab.sum() > 5 else np.nan,
        "stab_mae": float(np.abs(pred[stab] - y[stab]).mean()) if stab.any() else np.nan,
        "auc_stab": float(roc_auc_score(stab, -pred)) if 0 < stab.sum() < len(y) else np.nan,
        "detpr30": float(stab[np.argsort(pred)[:30]].mean()),
        "ndcg30": ndcg_stab(y, pred),
    }


def load(exp):
    """features_ablation (wtz/mtz/...) joined with features_bio (cw/far/site/bio).

    S669 carries 17 variants twice (repeat measurements from different sources:
    same wt_id + mutation, different ddG). Their *features* are byte-identical in
    both tables — verified — since the features depend only on (wt_key, position,
    wt_aa, mut_aa). So the bio side is deduplicated and joined many-to-one, which
    keeps both measurements as separate labelled rows sharing one feature vector.
    """
    proc = ROOT / "data/processed" / exp
    df = pd.read_parquet(proc / "features_ablation.parquet")
    n0 = len(df)
    for name in ("features_bio.parquet", "features_msa.parquet"):
        path = proc / name
        if not path.exists():       # features_msa is absent until the MSAs are fetched
            continue
        side = (pd.read_parquet(path)
                .drop(columns=["ddg"])
                .drop_duplicates(subset=["wt_id", "mutation"]))
        df = df.merge(side, on=["wt_id", "mutation"], how="inner",
                      validate="many_to_one")
        assert len(df) == n0, f"{exp}: {name} join lost {n0 - len(df)} rows"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS))
    ap.add_argument("--out", default="results.csv")
    args = ap.parse_args()

    tsu = load("tsuboyama_bench_fast")
    y = tsu["ddg"].to_numpy(float)
    groups = tsu["wt_id"].to_numpy()
    s669 = load("s669")
    ys = -s669["ddg"].to_numpy(float)     # S669 sign convention is inverted (results/13)
    print(f"Tsuboyama {len(tsu)} muts / {tsu.wt_id.nunique()} proteins; "
          f"stabilizing {(y < STAB).sum()} ({(y < STAB).mean():.1%})")
    print(f"S669 {len(s669)} variants; stabilizing {(ys < STAB).sum()}\n", flush=True)

    rows, preds = [], {}
    for cfg in args.configs:
        cols, n_inv, n_side = assemble(CONFIGS[cfg])
        X = tsu[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        Xs = s669[cols].replace([np.inf, -np.inf], np.nan).to_numpy(float)
        t0 = time.time()
        oof = np.full(len(y), np.nan)
        s_pred = []
        for fi, (tr, te) in enumerate(GroupKFold(n_splits=K).split(X, y, groups), 1):
            Xa, ya = augment(X[tr], y[tr], n_inv, n_side)
            model = make_model("mlp")
            model.named_steps["est"].n_jobs = N_JOBS
            model.fit(Xa, ya)
            oof[te] = model.predict(X[te])
            s_pred.append(model.predict(Xs))
            print(f"  [{cfg}] fold {fi}/{K} done ({time.time() - t0:.0f}s)", flush=True)
        s_mean = np.mean(s_pred, axis=0)
        preds[cfg] = oof
        for tag, yy, pp in (("tsu_oof", y, oof), ("s669", ys, s_mean)):
            m = metrics(yy, pp, tag, cfg)
            m["dims"] = len(cols)
            m["secs"] = round(time.time() - t0)
            rows.append(m)
        a, b = rows[-2], rows[-1]
        print(f"{cfg:14s} dims={len(cols):4d}  OOF r={a['r']:.3f} rho={a['rho']:.3f} "
              f"MAE={a['mae']:.3f} | stab rho={a['stab_rho']:.3f} bias={a['stab_bias']:+.3f} "
              f"AUC={a['auc_stab']:.3f} | S669 r={b['r']:.3f} rho={b['rho']:.3f}\n", flush=True)

        pd.DataFrame(rows).to_csv(OUT / args.out, index=False)
        # name the OOF dump after --out so separate invocations don't clobber
        # each other's predictions (the class-breakdown analyses read these back)
        stem = Path(args.out).stem
        pd.DataFrame({"wt_id": tsu.wt_id, "mutation": tsu.mutation, "ddg": y,
                      **preds}).to_csv(
            ROOT / f"data/processed/_analysis/exp14_oof_{stem}.csv", index=False)

    res = pd.DataFrame(rows)
    print("\n=== held-out Tsuboyama (GroupKFold on protein) ===")
    print(res[res.set == "tsu_oof"][
        ["config", "dims", "r", "rho", "mae", "stab_rho", "stab_bias",
         "auc_stab", "detpr30", "ndcg30"]].round(3).to_string(index=False))
    print("\n=== S669 transfer ===")
    print(res[res.set == "s669"][
        ["config", "r", "rho", "mae", "stab_rho", "auc_stab"]].round(3).to_string(index=False))
    print(f"\nwrote {OUT / args.out}")


if __name__ == "__main__":
    main()
