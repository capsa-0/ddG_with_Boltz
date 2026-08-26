"""
Phase 3b — does our Boltz ΔΔG predict MAVE fitness better than Rosetta's?

Two layers, both on the same 13 Tier-1 datasets and the same variants:

  Layer 1 (direct, their Figure 2A): per dataset, Spearman between s_exp and each
    predictor on its own -- our ΔΔG, Rosetta ΔΔG, GEMME ΔΔE. No model, no fitting.

  Layer 2 (RF4Mave, their Figure 2B): the leave-one-protein-out random forest from
    rf4mave.py, run twice per feature set -- once with Rosetta's ΔΔG and once with
    ours, everything else identical. The paired difference is the result; their
    published median over all 39 datasets is context, not the target, because we
    cover 13.

Both layers are reported full and with UBI4 dropped: ubiquitin is the one Tier-1
protein homologous to Tsuboyama (see homology/mave_le200_leakage.csv), so our regimes
have effectively seen it and Rosetta has not.

Sign: destabilizing (high ΔΔG) means low fitness, so the raw correlation is negative
by construction. We check that before reading anything into the magnitudes, so a sign
error cannot masquerade as a result.

    conda run -n ddG_with_Boltz python results/15_mave_stability_transfer/score.py
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from ddg.evaluation.metrics import compute_metrics  # noqa: E402
from build_corpus import load_tier, MAX_LEN, SRC_DIR  # noqa: E402
from rf4mave import run_lopo, FEATURE_SETS, MISSING  # noqa: E402

AA = "ACDEFGHIKLMNPQRSTVWY"
PRED_PATH = ROOT / "data/processed/mave_hoie_le200/mave_ddg_predictions.csv"
SUB_MATRIX = ROOT / "data/raw/mave_hoie/mut_matrix_alphabetical.npy"
REGIMES = ("A_tsuboyama", "B_fireprot", "D_finetuned", "mean")
SINGLE = r"[A-Z]\d+[A-Z]"


def _grid(values: dict, n_pos: int) -> np.ndarray:
    """(n_pos, 20) grid of a per-variant score, NaN where absent."""
    g = np.full((n_pos, 20), np.nan, dtype=np.float32)
    idx = {a: i for i, a in enumerate(AA)}
    for (pos, mut), v in values.items():
        if 1 <= pos <= n_pos and mut in idx and np.isfinite(v):
            g[pos - 1, idx[mut]] = v
    return g


def build_frames(by_protein, preds, ddg_source, sub_matrix):
    """One feature frame per dataset, with the column names rf4mave expects.

    `ddg_source` is either "rosetta" or a column of `preds` -- everything else in the
    frame is identical between the two arms, so the LOPO comparison is paired.
    """
    frames, names, proteins = [], [], []
    idx = {a: i for i, a in enumerate(AA)}
    for protein, entry in sorted(by_protein.items()):
        seq = entry["meta"]["sequence"]
        L = len(seq)

        if ddg_source == "rosetta":
            src = None
        else:
            p = preds[preds["wt_id"] == protein]
            src = {(int(m[1:-1]), m[-1]): v
                   for m, v in zip(p["mutation"], p[ddg_source])}

        for dataset, df in sorted(entry["datasets"].items()):
            d = df[df["variant"].astype(str).str.fullmatch(SINGLE, na=False)]
            pos_all = d["variant"].str[1:-1].astype(int)
            mut_all = d["variant"].str[-1]

            gemme_g = _grid(dict(zip(zip(pos_all, mut_all), d["gemme_score_01"])), L)
            if src is None:
                ddg_g = _grid(dict(zip(zip(pos_all, mut_all),
                                       d["Rosetta_ddg_score_02"])), L)
            else:
                ddg_g = _grid(src, L)

            scored = d[d["score_00"].notna()]
            if scored.empty:
                continue
            pos = scored["variant"].str[1:-1].astype(int).to_numpy()
            wt = scored["variant"].str[0].to_numpy()
            mut = scored["variant"].str[-1].to_numpy()
            mi = np.array([idx[m] for m in mut])
            wi = np.array([idx[w] for w in wt])

            out = {}
            # Rank-normalise s_exp to [0,1], their protocol.
            s = scored["score_00"].to_numpy(dtype=float)
            out["score"] = (pd.Series(s).rank(method="average").to_numpy() - 1) \
                / max(len(s) - 1, 1)

            for tag, g in (("ros", ddg_g), ("gemme", gemme_g)):
                block = g[pos - 1]                       # (n, 20)
                for a, j in idx.items():
                    out[f"{tag}_aa_p0_{a}"] = block[:, j]
                out[f"{tag}_aa_wt_p"] = block[np.arange(len(pos)), mi]
                with warnings.catch_warnings():
                    # A position with no value at all for any of the 20 substitutions
                    # gives an all-NaN slice; NaN is the right answer and becomes the
                    # -100 sentinel below, same as theirs.
                    warnings.simplefilter("ignore", RuntimeWarning)
                    out[f"{tag}_M_p0"] = np.nanmedian(block, axis=1)

            out["mave_wt_to_mut"] = sub_matrix[wi, mi]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out["mave_wt_to_any"] = np.nanmean(sub_matrix[wi, :], axis=1)
                out["mave_any_to_mut"] = np.nanmean(sub_matrix[:, mi], axis=0)

            frame = pd.DataFrame(out).astype(np.float32).fillna(MISSING)
            frames.append(frame)
            names.append(dataset)
            proteins.append(protein)
    return frames, names, proteins


def layer1(by_protein, preds):
    """Direct per-dataset Spearman for each predictor, no model."""
    rows = []
    for protein, entry in sorted(by_protein.items()):
        p = preds[preds["wt_id"] == protein].set_index("mutation")
        for dataset, df in sorted(entry["datasets"].items()):
            d = df[df["variant"].astype(str).str.fullmatch(SINGLE, na=False)]
            d = d[d["score_00"].notna()].set_index("variant")
            joined = d.join(p, how="inner")
            base = dict(dataset=dataset, protein=protein, n=len(joined))
            y = joined["score_00"].to_numpy(float)
            pairs = ([("rosetta", "Rosetta_ddg_score_02"),
                      ("gemme", "gemme_score_01")]
                     + [(f"boltz_{r}", f"ddg_{r}") for r in REGIMES])
            for label, col in pairs:
                if col not in joined.columns:
                    continue
                x = joined[col].to_numpy(float)
                m = np.isfinite(x) & np.isfinite(y)
                r = compute_metrics(y[m], x[m])["spearman"] if m.sum() > 10 else np.nan
                base[f"rho_{label}"] = r
            rows.append(base)
    return pd.DataFrame(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Score the MAVE transfer benchmark")
    ap.add_argument("--preds", type=Path, default=PRED_PATH)
    ap.add_argument("--src", type=Path, default=SRC_DIR)
    ap.add_argument("--regime", default="mean", choices=REGIMES)
    ap.add_argument("--out", type=Path, default=HERE)
    ap.add_argument("--skip-lopo", action="store_true")
    args = ap.parse_args(argv)

    if not args.preds.exists():
        print(f"ERROR: {args.preds} not found — run predict_ddg.py first.",
              file=sys.stderr)
        return 1

    by_protein = load_tier(args.src, MAX_LEN)
    preds = pd.read_csv(args.preds)
    sub_matrix = np.load(SUB_MATRIX)
    print(f"{len(by_protein)} proteins, "
          f"{sum(len(e['datasets']) for e in by_protein.values())} datasets; "
          f"{len(preds):,} ddG predictions")

    # ---- Layer 1: direct correlations ----
    l1 = layer1(by_protein, preds)
    l1.to_csv(args.out / "layer1_direct.csv", index=False)
    print("\n=== Layer 1: direct Spearman vs s_exp (signed) ===")
    print(l1.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
    print("\nmedian signed rho:")
    for c in [c for c in l1.columns if c.startswith("rho_")]:
        full = l1[c].median()
        clean = l1[l1["protein"] != "UBI4"][c].median()
        print(f"  {c:18} full {full:+.3f}   UBI4-dropped {clean:+.3f}")
    boltz = f"rho_boltz_{args.regime}"
    if l1[boltz].median() > 0:
        print(f"\nWARNING: median {boltz} is POSITIVE. Destabilizing ddG should mean "
              f"LOW fitness — check the sign convention before reading anything into "
              f"these numbers.", file=sys.stderr)

    if args.skip_lopo:
        return 0

    # ---- Layer 2: paired LOPO random forests ----
    print("\n=== Layer 2: leave-one-protein-out RF ===")
    rows, per_ds = [], []
    arms = [("rosetta", "rosetta"), ("boltz", f"ddg_{args.regime}")]
    for model in ("null_smave", "dde_only", "ddg_only", "ddg_dde",
                  "position_context"):
        for arm, source in arms:
            # null and ddE-only do not use the stability column: run once.
            shared = model in ("null_smave", "dde_only")
            if shared and arm != "rosetta":
                continue
            frames, names, proteins = build_frames(by_protein, preds, source,
                                                   sub_matrix)
            res, cols_used = run_lopo(frames, names, proteins,
                                      FEATURE_SETS[model], verbose=False)
            res.insert(0, "arm", "shared" if shared else arm)
            res.insert(0, "model", model)
            per_ds.append(res)
            clean = res[res["protein"] != "UBI4"]["spearman"].median()
            rows.append(dict(model=model, arm=res["arm"].iloc[0],
                             n_features=len(cols_used),
                             median_spearman=res["spearman"].median(),
                             median_spearman_no_ubi4=clean))
            print(f"  {model:18} {res['arm'].iloc[0]:8} "
                  f"median rho = {res['spearman'].median():+.3f}  "
                  f"(UBI4-dropped {clean:+.3f})", flush=True)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.out / "layer2_lopo_summary.csv", index=False)
    pd.concat(per_ds).to_csv(args.out / "layer2_lopo_per_dataset.csv", index=False)
    print("\n" + summary.to_string(index=False))
    print(f"\nwrote {args.out}/layer1_direct.csv, layer2_lopo_*.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
