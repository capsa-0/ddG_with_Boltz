"""
Phase 3a — predict ΔΔG for every mutation in the Tier-1 MAVE corpus.

Trains the same three regimes used in results/08 and results/09, then applies them to
the 25,213 MAVE mutations:

  A. Tsuboyama-only : fit on ALL Tsuboyama.
  B. FireProt-only  : fit on ALL FireProt (its own imputer/scaler).
  D. Fine-tuned     : pretrain on Tsuboyama, warm-start continue on FireProt,
                      reusing Tsuboyama's transform so the pretrained scaler stays valid.

All three use the 256-dim concat `wtz|mtz` representation, antisymmetry augmentation
and a 5-seed MLP ensemble — the recipe adopted in results/07.

Deliberately NOT `ddg.scan.predict`: that assumes a single protein (it pivots on
position and reads `wt_id.iloc[0]`), and this corpus has 11.

Sign convention: positive ΔΔG = destabilizing, inherited from the training corpora.
There is no ΔΔG label here to auto-flip against — `score.py` handles the sign when it
correlates against MAVE fitness, where the expected relationship is negative.

    conda run -n ddG_with_Boltz python results/15_mave_stability_transfer/predict_ddg.py

Requires (built on the cluster, then rsync'd back):
    data/processed/mave_hoie_le200/features_summary.parquet
"""
import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

Z, N_SEED = 128, 5
# Feature sets. `concat` is the results/07 convention this experiment was published with.
# `diag` is the pair-track diagonal alone: results/14 found it matches every 256-d
# construction on blind transfer at half the width, while the pooled LEVELS (wtz/mtz)
# carry a corpus-specific per-protein offset that costs -0.141 r across corpora. Since
# MAVE fitness is itself a transfer task, `diag` is the default here.
FEATURE_SETS = {
    "concat": [f"{p}_{j}" for p in ("wtz", "mtz") for j in range(Z)],
    "diag": [f"zdiag_{j}" for j in range(Z)],
}
FEAT = FEATURE_SETS["diag"]        # rebound in main() from --features
TSU_PATH = ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet"
FP_PATH = ROOT / "data/processed/fireprot_le500/features_ablation.parquet"
MAVE_PATH = ROOT / "data/processed/mave_hoie_le200/features_summary.parquet"
OUT_PATH = ROOT / "data/processed/mave_hoie_le200/mave_ddg_predictions.csv"
REGIMES = ("A_tsuboyama", "B_fireprot", "D_finetuned")


def mat(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in FEAT if c not in df.columns]
    if missing:
        raise ValueError(
            f"features table is missing {len(missing)} column(s) "
            f"(e.g. {missing[:3]}). Rebuild it with "
            f"`feature.blocks: [zdiag, zpool, wtz, mtz]` in the experiment YAML.")
    return df[FEAT].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def augment(X, y):
    """Antisymmetry, dispatched on the feature form.

    concat: the reverse mutation swaps the [wtz|mtz] halves.
    diag:   zdiag is a DIFFERENCE (mut - wt), so the reverse mutation negates it.
    Applying the half-swap to a difference block is meaningless — that bug is why
    `dz`+augmentation had never been computable before results/14.
    """
    if FEAT is FEATURE_SETS["diag"] or len(FEAT) == Z:
        return np.vstack([X, -X]), np.concatenate([y, -y])
    swapped = np.concatenate([X[:, Z:], X[:, :Z]], axis=1)
    return np.vstack([X, swapped]), np.concatenate([y, -y])


LEGACY_ESTIMATOR = False   # set by --estimator legacy


def members():
    """The project-default estimator (ddg.evaluation.models.make_model('mlp')).

    CORRECTED 2026-08-27. This previously used `max_iter=250, early_stopping=False`,
    the same defect found in results/09, where it cost regime A ~0.16 Pearson on S669:
    `max_iter` counts EPOCHS, so the regime with the most data over-trains hardest.
    `--estimator legacy` restores the old settings to reproduce published numbers.
    """
    if LEGACY_ESTIMATOR:
        return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                             batch_size=256, max_iter=250, early_stopping=False,
                             random_state=s, warm_start=True) for s in range(N_SEED)]
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=1000, early_stopping=True,
                         n_iter_no_change=25, validation_fraction=0.1,
                         random_state=s, warm_start=True) for s in range(N_SEED)]


def fit_regimes(tsu: pd.DataFrame, fp: pd.DataFrame):
    Xt, yt = augment(mat(tsu), tsu["ddg"].to_numpy(float))
    impA = SimpleImputer(strategy="median").fit(Xt)
    scaA = StandardScaler().fit(impA.transform(Xt))
    TA = lambda X: scaA.transform(impA.transform(X))  # noqa: E731
    A = members()
    for m in A:
        m.fit(TA(Xt), yt)

    Xf, yf = augment(mat(fp), fp["ddg"].to_numpy(float))
    impB = SimpleImputer(strategy="median").fit(Xf)
    scaB = StandardScaler().fit(impB.transform(Xf))
    TB = lambda X: scaB.transform(impB.transform(X))  # noqa: E731
    B = members()
    for m in B:
        m.fit(TB(Xf), yf)

    D = copy.deepcopy(A)
    for m in D:
        m.set_params(learning_rate_init=1e-3, max_iter=400)
        m.fit(TA(Xf), yf)

    return [("A_tsuboyama", A, TA), ("B_fireprot", B, TB), ("D_finetuned", D, TA)]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Predict ΔΔG for the MAVE corpus")
    ap.add_argument("--mave", type=Path, default=MAVE_PATH)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--features", default="diag", choices=sorted(FEATURE_SETS),
                    help="pair-track readout (default: diag, per results/14)")
    ap.add_argument("--estimator", default="default", choices=("default", "legacy"),
                    help="'legacy' reproduces the pre-2026-08-27 overfit settings")
    args = ap.parse_args(argv)

    global FEAT, LEGACY_ESTIMATOR
    FEAT = FEATURE_SETS[args.features]
    LEGACY_ESTIMATOR = args.estimator == "legacy"
    print(f"features: {args.features} ({len(FEAT)}d) | estimator: {args.estimator}")

    if not args.mave.exists():
        print(f"ERROR: {args.mave} not found — run the cluster pipeline first "
              f"(./slurm/submit_scan.sh experiment_configs/mave_hoie_le200.yaml 256 3) "
              f"and rsync data/processed/mave_hoie_le200/ back.", file=sys.stderr)
        return 1

    tsu = pd.read_parquet(TSU_PATH)
    fp = pd.read_parquet(FP_PATH)
    print(f"train: Tsuboyama {len(tsu):,} muts / {tsu.wt_id.nunique()} proteins; "
          f"FireProt {len(fp):,} muts / {fp.wt_id.nunique()} proteins")

    mave = pd.read_parquet(args.mave)
    print(f"scan : {len(mave):,} mutations / {mave.wt_id.nunique()} proteins")

    regimes = fit_regimes(tsu, fp)
    X = mat(mave)
    out = mave[["wt_id", "mutation"]].copy()
    for name, ms, T in regimes:
        preds = np.stack([m.predict(T(X)) for m in ms])
        out[f"ddg_{name}"] = preds.mean(axis=0)
        out[f"ddg_{name}_seed_sd"] = preds.std(axis=0)
        print(f"  {name:14} mean {preds.mean():+.3f}  sd {preds.mean(axis=0).std():.3f}")
    cols = [f"ddg_{r}" for r in REGIMES]
    out["ddg_mean"] = out[cols].mean(axis=1)
    out["ddg_regime_sd"] = out[cols].std(axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
