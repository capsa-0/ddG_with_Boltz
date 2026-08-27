"""Condense the in-distribution class breakdown into one committed table.

`tsu_class_error.py` writes a per-variant table (`tsu_mut_classes.csv`) into the
gitignored `data/processed/_analysis/`. This script reduces it to the class-level
summary the README and `build_report.py` quote, so those numbers live in this folder
and cannot drift from a re-run.

Same conventions as `transfer_class_error.py`: ddG POSITIVE = destabilizing, error is
`pred - true`, reported raw and protein-centred, and every class MAE is read against
that class's own sd of true ddG.

    python results/12_error_anatomy/indist_class_tables.py

Writes `indist_class_tables.csv`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
HERE = ROOT / "results/12_error_anatomy"
SRC = ROOT / "data/processed/_analysis/tsu_mut_classes.csv"


def stats(g: pd.DataFrame, pred: str = "pred_OOF") -> dict:
    y, e, ec = g.ddg.values, g.err.values, g.err_c.values
    sd = y.std(ddof=1) if len(y) > 1 else np.nan
    return dict(n=len(g), MAE=np.abs(e).mean(), MAE_centred=np.abs(ec).mean(),
                bias=e.mean(), sd_true=sd,
                MAE_sd=np.abs(e).mean() / sd if sd and sd > 0 else np.nan,
                rho=spearmanr(g[pred], y).correlation if len(y) >= 25 else np.nan)


def main() -> None:
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found — run tsu_class_error.py first")
    d = pd.read_csv(SRC)
    rows = [dict(grouping="overall", klass="all", **stats(d))]

    for grouping, col in [("burial", "burial_q"), ("gly_pro", "cls"),
                          ("direction", "dir"), ("volume_change", "volq"),
                          ("wt_aa", "wt_aa"), ("mut_aa", "mut_aa")]:
        for k, g in d.groupby(col):
            rows.append(dict(grouping=grouping, klass=str(k), **stats(g)))

    # The one interaction cell results/10 predicted and this folder confirmed.
    buried = d[d.burial_q == "buried"]
    for k, m in [("buried, from Gly", buried.wt_aa == "G"),
                 ("buried, not Gly", buried.wt_aa != "G")]:
        rows.append(dict(grouping="burial_x_gly", klass=k, **stats(buried[m])))

    out = pd.DataFrame(rows)
    out.to_csv(HERE / "indist_class_tables.csv", index=False)
    print(out.round(3).to_string(index=False))
    print(f"\nWrote {HERE / 'indist_class_tables.csv'}")


if __name__ == "__main__":
    main()
