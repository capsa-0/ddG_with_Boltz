"""
Module: predict
Description: Score a full mutational scan with the project's ΔΔG regressor.

A scan has no labels, so the model has to come from somewhere else: it is fit on
the labelled corpora and applied, without refitting, to every mutation of the
scanned protein. This is the same machinery as results/09_external_benchmarks —
**concat features** (``wtz`` + ``mtz``), **antisymmetry augmentation**, a **5-seed
MLP ensemble** — under the three training regimes benchmarked there:

  A — Tsuboyama-only : fit on all of the Tsuboyama corpus.
  B — FireProt-only  : fit on all of FireProt (its own imputer/scaler).
  D — fine-tuned     : pretrain on Tsuboyama, warm-start continue on FireProt.

All three are reported per mutation, plus their mean and spread. The spread is the
honest uncertainty signal here: the regimes differ only in *training distribution*,
so where they disagree the prediction is extrapolating beyond what any corpus
covers. (On the diverse S669 benchmark these reached per-protein median r of
0.46 / 0.56 / 0.61 respectively — see results/09.)

Sign convention: every training corpus uses **positive = destabilizing**, and the
predictions inherit it. Unlike ddg.evaluation.transfer there are no measured labels
to detect a flipped convention against, so nothing is auto-flipped.

    python -m ddg.scan predict --config experiment_configs/scan_GLA_human.yaml
"""

import argparse
import copy
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from ddg.scan.mutations import AA_ORDER

logger = logging.getLogger(__name__)

Z_DIM = 128
N_SEEDS = 5
# The concat representation adopted in results/07 and used by 08/09.
FEATURES = [f"{block}_{j}" for block in ("wtz", "mtz") for j in range(Z_DIM)]

REGIMES = ("A_tsuboyama", "B_fireprot", "D_finetuned")
REGIME_LABEL = {
    "A_tsuboyama": "A — Tsuboyama-only",
    "B_fireprot": "B — FireProt-only",
    "D_finetuned": "D — Tsuboyama→FireProt fine-tuned",
}

DEFAULT_TSUBOYAMA = "data/processed/tsuboyama_bench_fast/features_ablation.parquet"
DEFAULT_FIREPROT = "data/processed/fireprot_le500/features_ablation.parquet"


# --------------------------------------------------------------------------- #
# Model (identical hyperparameters to results/09_external_benchmarks)
# --------------------------------------------------------------------------- #
def _members() -> list[MLPRegressor]:
    """The 5 seed-decorrelated MLPs whose predictions get averaged."""
    return [MLPRegressor((256, 128, 64), alpha=3e-3, learning_rate_init=1e-3,
                         batch_size=256, max_iter=250, early_stopping=False,
                         random_state=seed, warm_start=True)
            for seed in range(N_SEEDS)]


def _matrix(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix on the fixed concat column set; inf -> NaN for the imputer."""
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(
            f"features table is missing {len(missing)} concat column(s) "
            f"(e.g. {missing[:3]}). Rebuild it with "
            f"`feature.blocks: [zdiag, zpool, wtz, mtz]` in the experiment YAML.")
    return df[FEATURES].replace([np.inf, -np.inf], np.nan).to_numpy(float)


def _augment(X: np.ndarray, y: np.ndarray):
    """Antisymmetry for concat features: swap [wtz|mtz] halves, negate ΔΔG.

    ΔΔG(A→B) = −ΔΔG(B→A), and for this representation the reverse mutation is just
    the two halves exchanged — so the identity is expressible as an input transform
    and doubles the training set for free.
    """
    swapped = np.concatenate([X[:, Z_DIM:], X[:, :Z_DIM]], axis=1)
    return np.vstack([X, swapped]), np.concatenate([y, -y])


def _fit_transform_stack(X: np.ndarray):
    """Fit median-impute -> standardize on X; return the transform callable."""
    imputer = SimpleImputer(strategy="median").fit(X)
    scaler = StandardScaler().fit(imputer.transform(X))
    return lambda M: scaler.transform(imputer.transform(M))


def train_regimes(tsuboyama: pd.DataFrame, fireprot: pd.DataFrame) -> dict:
    """Fit regimes A, B and D. Returns {regime: (members, transform)}."""
    Xt, yt = _augment(_matrix(tsuboyama), tsuboyama["ddg"].to_numpy(float))
    transform_t = _fit_transform_stack(Xt)
    logger.info("regime A: fitting %d MLPs on Tsuboyama (%d rows augmented)",
                N_SEEDS, len(Xt))
    regime_a = _members()
    for model in regime_a:
        model.fit(transform_t(Xt), yt)

    Xf, yf = _augment(_matrix(fireprot), fireprot["ddg"].to_numpy(float))
    transform_f = _fit_transform_stack(Xf)
    logger.info("regime B: fitting %d MLPs on FireProt (%d rows augmented)",
                N_SEEDS, len(Xf))
    regime_b = _members()
    for model in regime_b:
        model.fit(transform_f(Xf), yf)

    # Fine-tune: warm-start A's weights on FireProt, reusing Tsuboyama's transform
    # so the pretrained scaler stays valid for the continued training.
    logger.info("regime D: warm-starting A on FireProt")
    regime_d = copy.deepcopy(regime_a)
    for model in regime_d:
        model.set_params(learning_rate_init=1e-3, max_iter=400)
        model.fit(transform_t(Xf), yf)

    return {
        "A_tsuboyama": (regime_a, transform_t),
        "B_fireprot": (regime_b, transform_f),
        "D_finetuned": (regime_d, transform_t),
    }


def _predict_members(members, transform, X: np.ndarray):
    """Return (mean over seeds, SD over seeds) for one regime."""
    per_seed = np.vstack([m.predict(transform(X)) for m in members])
    return per_seed.mean(axis=0), per_seed.std(axis=0)


# --------------------------------------------------------------------------- #
# Scan scoring
# --------------------------------------------------------------------------- #
def _parse_mutation(mutation: str) -> tuple[str, int, str]:
    text = str(mutation)
    return text[0], int(text[1:-1]), text[-1]


def score_scan(scan: pd.DataFrame, tsuboyama: pd.DataFrame, fireprot: pd.DataFrame,
               first_residue: int = 1) -> pd.DataFrame:
    """
    Predict ΔΔG for every mutation in ``scan`` under all three regimes.

    The pipeline numbers mutations 1-based over the sequence it was given
    (``ddg.datasets.prepare`` validates the wild-type residue that way).
    ``first_residue`` is the number that residue 1 carries in the numbering results
    should be *reported* in, so a mature chain can be labelled in its precursor's
    numbering and join directly against external tables. The reported
    ``mutation``/``position`` use it; ``mutation_local``/``position_local`` keep the
    pipeline's own 1-based labels, which are the keys back into the slim store.

    Returns one row per mutation with a ``ddg_<regime>`` column per regime, the
    per-regime seed SD, and the across-regime mean/SD.
    """
    models = train_regimes(tsuboyama, fireprot)
    X = _matrix(scan)
    logger.info("scoring %d scan mutations (reported from residue %d)",
                len(X), first_residue)

    parsed = [_parse_mutation(m) for m in scan["mutation"]]
    offset = int(first_residue) - 1
    wt_aa = [w for w, _, _ in parsed]
    mut_aa = [m for _, _, m in parsed]
    local_pos = [p for _, p, _ in parsed]
    position = [p + offset for p in local_pos]
    out = pd.DataFrame({
        "wt_id": scan["wt_id"].to_numpy(),
        "mutation": [f"{w}{p}{m}" for w, p, m in zip(wt_aa, position, mut_aa)],
        "position": position,
        "wt_aa": wt_aa,
        "mut_aa": mut_aa,
        "mutation_local": scan["mutation"].to_numpy(),
        "position_local": local_pos,
    })

    for regime, (members, transform) in models.items():
        mean, sd = _predict_members(members, transform, X)
        out[f"ddg_{regime}"] = mean
        out[f"ddg_{regime}_seed_sd"] = sd

    stacked = np.vstack([out[f"ddg_{r}"].to_numpy() for r in REGIMES])
    out["ddg_mean"] = stacked.mean(axis=0)
    out["ddg_regime_sd"] = stacked.std(axis=0)
    return out.sort_values(["position", "mut_aa"]).reset_index(drop=True)


def scan_matrix(predictions: pd.DataFrame, column: str) -> pd.DataFrame:
    """Pivot one prediction column into a position x target-residue matrix.

    Rows are positions in order, columns are the 20 residues in AA_ORDER; the
    wild-type residue's own cell is NaN (no such mutation exists).
    """
    matrix = predictions.pivot(index="position", columns="mut_aa", values=column)
    matrix = matrix.reindex(columns=list(AA_ORDER)).sort_index()
    wt_by_pos = predictions.drop_duplicates("position").set_index("position")["wt_aa"]
    matrix.insert(0, "wt_aa", wt_by_pos.reindex(matrix.index))
    return matrix


def _summary(predictions: pd.DataFrame, top_n: int = 25) -> dict:
    """Headline statistics + the most/least destabilizing mutations."""
    cols = ["mutation", "position", "wt_aa", "mut_aa",
            *[f"ddg_{r}" for r in REGIMES], "ddg_mean", "ddg_regime_sd"]
    cols = [c for c in cols if c in predictions.columns]
    ranked = predictions.sort_values("ddg_mean")
    per_regime = {
        r: {
            "mean": float(predictions[f"ddg_{r}"].mean()),
            "sd": float(predictions[f"ddg_{r}"].std()),
            "min": float(predictions[f"ddg_{r}"].min()),
            "max": float(predictions[f"ddg_{r}"].max()),
        } for r in REGIMES
    }
    corr = {
        f"{a}_vs_{b}": float(predictions[f"ddg_{a}"].corr(predictions[f"ddg_{b}"]))
        for i, a in enumerate(REGIMES) for b in REGIMES[i + 1:]
    }
    return {
        "wt_id": str(predictions["wt_id"].iloc[0]),
        "n_mutations": int(len(predictions)),
        "n_positions": int(predictions["position"].nunique()),
        "sign_convention": "positive ΔΔG = destabilizing",
        "per_regime": per_regime,
        "regime_agreement_pearson": corr,
        "mean_regime_sd": float(predictions["ddg_regime_sd"].mean()),
        "most_destabilizing": ranked.tail(top_n)[cols].iloc[::-1].to_dict("records"),
        "most_stabilizing": ranked.head(top_n)[cols].to_dict("records"),
    }


def _load(path) -> pd.DataFrame:
    path = str(path)
    return pd.read_csv(path) if path.endswith(".csv") else pd.read_parquet(path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ddg.scan predict",
        description="Predict ΔΔG for every mutation of a full scan (regimes A/B/D)")
    ap.add_argument("--config", help="scan experiment YAML (locates the features table)")
    ap.add_argument("--scan", help="scan features table (overrides --config)")
    ap.add_argument("--out", help="output dir (default <processed>/scan)")
    ap.add_argument("--tsuboyama", default=DEFAULT_TSUBOYAMA,
                    help="Tsuboyama features table (regimes A and D)")
    ap.add_argument("--fireprot", default=DEFAULT_FIREPROT,
                    help="FireProt features table (regimes B and D)")
    ap.add_argument("--first-residue", type=int,
                    help="number of the sequence's first residue in the reported "
                         "numbering; default: the scan config's scan.first_residue, "
                         "else 1")
    ap.add_argument("--names-config", default="ddg/config/internal_config.yaml")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    first_residue = args.first_residue
    if args.config:
        from ddg.config.config_loader import ProjectConfig
        config = ProjectConfig(experiment_yaml_path=args.config,
                               internal_yaml_path=args.names_config)
        processed = Path(config.exp_processed_dir)
        scan_path = Path(args.scan) if args.scan else processed / "features_summary.parquet"
        out = Path(args.out) if args.out else processed / "scan"
        if first_residue is None:
            first_residue = config.exp_config.get("scan", {}).get("first_residue", 1)
    elif args.scan:
        scan_path = Path(args.scan)
        out = Path(args.out) if args.out else scan_path.parent / "scan"
    else:
        raise SystemExit("provide --config or --scan")
    first_residue = 1 if first_residue is None else int(first_residue)

    for path, what in ((scan_path, "scan features"),
                       (Path(args.tsuboyama), "Tsuboyama features"),
                       (Path(args.fireprot), "FireProt features")):
        if not path.exists():
            raise SystemExit(f"{what} table not found: {path}")

    scan = _load(scan_path)
    tsuboyama, fireprot = _load(args.tsuboyama), _load(args.fireprot)
    logger.info("scan %d rows | train: Tsuboyama %d, FireProt %d",
                len(scan), len(tsuboyama), len(fireprot))

    predictions = score_scan(scan, tsuboyama, fireprot, first_residue=first_residue)
    out.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(out / "scan_predictions.csv", index=False)
    for regime in (*REGIMES, "mean"):
        column = f"ddg_{regime}" if regime != "mean" else "ddg_mean"
        scan_matrix(predictions, column).to_csv(out / f"scan_matrix_{regime}.csv")

    summary = _summary(predictions)
    summary["scan_features"] = str(scan_path)
    summary["train_tsuboyama"] = str(args.tsuboyama)
    summary["train_fireprot"] = str(args.fireprot)
    summary["first_residue"] = first_residue
    (out / "scan_summary.json").write_text(json.dumps(summary, indent=2))

    if not args.no_figures:
        from ddg.scan import plots
        made = plots.make_all(predictions, out / "figures")
        logger.info("wrote %d figures", len(made))

    print(f"\n=== scan {summary['wt_id']}: {summary['n_mutations']} mutations over "
          f"{summary['n_positions']} positions "
          f"({predictions.position.min()}-{predictions.position.max()}) ===")
    for regime in REGIMES:
        stats = summary["per_regime"][regime]
        print(f"{REGIME_LABEL[regime]:<38} mean {stats['mean']:+.2f}  "
              f"range [{stats['min']:+.2f}, {stats['max']:+.2f}] kcal/mol")
    print(f"regime agreement (Pearson): " +
          "  ".join(f"{k} {v:.3f}" for k, v in summary["regime_agreement_pearson"].items()))
    print(f"mean across-regime SD: {summary['mean_regime_sd']:.2f} kcal/mol")
    print(f"wrote -> {out}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    raise SystemExit(main())
