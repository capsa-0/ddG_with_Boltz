"""Score the GLA scan with the project's best *transfer* model (results/16).

results/10's original scan used `ddg.scan.predict`: concat features (`wtz`+`mtz`),
antisymmetry augmentation, three training regimes. results/14 and results/16 later
showed that readout is among the **worst** on blind transfer -- it loses to AFToolkit by
0.15 rho -- while the pair-track **diagonal alone** (`zdiag`, 128 d) is the best. This
rescores the scan with that model, under exactly the protocol results/16 benchmarked:

  * features  zdiag_0..127 (128 d), no augmentation
  * split     GroupKFold(5) on wt_id over the Tsuboyama corpus
  * model     make_model("mlp") -- impute -> scale -> 5-seed MLP(256,128,64) ensemble
  * transfer  the 5 fold models are averaged onto the scan

Emits `scan_predictions_diag.csv` with the same three columns as the FoldX table
(`ddg_varmed_by_mutation_foldx.csv`: mutation, position, ddg) so the two drop straight
into the same comparison, in the FoldX file's **global** numbering.

    python results/10_gla_scan/predict_scan_diag.py
"""
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ddg.evaluation.models import make_model

ROOT = Path(os.environ.get("DDG_ROOT", Path(__file__).resolve().parents[2]))
HERE = Path(__file__).resolve().parent
Z, K = 128, 5
COLS = [f"zdiag_{j}" for j in range(Z)]


def numbering_offset():
    """Scan corpus positions are local (mature chain); FoldX's are global. Recover the
    offset from results/10's own table, which carries both, and require it constant."""
    old = pd.read_csv(HERE / "scan_predictions_mean.csv")
    off = (old.position - old.position_local).unique()
    assert len(off) == 1, f"non-constant numbering offset: {off}"
    return int(off[0]), old


def main():
    tsu = pd.read_parquet(ROOT / "data/processed/tsuboyama_bench_fast/features_ablation.parquet")
    scan = pd.read_parquet(ROOT / "data/processed/scan_GLA_human/features_ablation.parquet")
    print(f"training corpus: {len(tsu)} mutations / {tsu.wt_id.nunique()} proteins")
    print(f"scan scored     : {len(scan)} mutations with embeddings")

    X, y, g = tsu[COLS].to_numpy(np.float32), tsu.ddg.to_numpy(float), tsu.wt_id.to_numpy()
    Xs = scan[COLS].to_numpy(np.float32)

    preds, t0 = [], time.time()
    for fi, (tr, _) in enumerate(GroupKFold(n_splits=K).split(X, y, g), 1):
        model = make_model("mlp")
        # run the 5-seed ensemble in parallel, as results/14 does; without this each
        # fold takes ~500 s instead of ~30 s. Parallelism only, so results are identical.
        model.named_steps["est"].n_jobs = -1
        model.fit(X[tr], y[tr])
        preds.append(model.predict(Xs))
        print(f"  fold {fi}/{K} done ({time.time() - t0:.0f}s)", flush=True)
    scan["ddg_pred"] = np.mean(preds, axis=0)
    scan["seed_sd"] = np.std(preds, axis=0)

    off, old = numbering_offset()
    scan["position_local"] = scan.mutation.str[1:-1].astype(int)
    scan["position"] = scan.position_local + off
    scan["mutation_global"] = (scan.mutation.str[0] + scan.position.astype(str)
                               + scan.mutation.str[-1])
    print(f"local -> global numbering offset: +{off}")

    # cross-check against the FoldX table: same mutation labels must exist there
    fx = pd.read_csv(HERE / "ddg_varmed_by_mutation_foldx.csv")
    shared = set(scan.mutation_global) & set(fx.mutation)
    print(f"mutations shared with the FoldX table: {len(shared)} "
          f"of {len(scan)} scored / {len(fx)} in FoldX")
    assert len(shared) > 0.9 * len(scan), "numbering mismatch against the FoldX table"

    out = (scan[["mutation_global", "position", "ddg_pred"]]
           .rename(columns={"mutation_global": "mutation", "ddg_pred": "ddg"})
           .sort_values("position", kind="stable"))
    p = HERE / "scan_predictions_diag.csv"
    out.to_csv(p, index=False)
    print(f"\nwrote {p}  ({len(out)} mutations, columns {list(out.columns)})")

    prev = set(old.mutation)
    print(f"superseding the previous {len(prev)}-mutation table: "
          f"{len(prev & set(out.mutation))} of them rescored, "
          f"{len(set(out.mutation) - prev)} newly covered")
    # a second view shaped for compare_foldx.py, which wants `mutation` in global
    # numbering plus wt_aa / mut_aa / position and a ddg_<regime> column
    full = scan[["wt_id", "mutation", "mutation_global", "position", "position_local",
                 "ddg_pred", "seed_sd"]].copy()
    full["wt_aa"] = full.mutation_global.str[0]
    full["mut_aa"] = full.mutation_global.str[-1]
    full.to_csv(HERE / "scan_predictions_diag_full.csv", index=False)
    (full.drop(columns=["mutation"])          # the local-numbering name; global wins
         .rename(columns={"mutation_global": "mutation", "ddg_pred": "ddg_diag"})
         [["wt_id", "mutation", "position", "wt_aa", "mut_aa", "ddg_diag", "seed_sd"]]
         .to_csv(HERE / "scan_predictions_diag_compare.csv", index=False))
    print(f"wrote {HERE/'scan_predictions_diag_compare.csv'} "
          f"(input for compare_foldx.py --regime diag)")


if __name__ == "__main__":
    main()
