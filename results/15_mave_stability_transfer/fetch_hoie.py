"""
Fetch the Høie et al. 2022 MAVE dataset collection.

Source: Høie, Cagiada, Frederiksen, Stein & Lindorff-Larsen (2022), Cell Reports
38:110207, "Predicting and interpreting large-scale mutagenesis data using analyses
of protein stability and conservation".  doi:10.1016/j.celrep.2021.110207
Data:   https://doi.org/10.5281/zenodo.5647207
        https://github.com/KULL-Centre/papers/tree/main/2021/ML-variants-Hoie-et-al

We take three things out of their 63 MB archive:

  * `data/preprocessed/*.txt` -- 39 merged PRISM tables, one per MAVE dataset, each
    carrying per variant  s_exp (MAVE fitness) | GEMME ddE | Rosetta ddG | DSSP ss,rsa
    plus the WT sequence in the header block.  This is what the Boltz corpus is
    built from.
  * `data/preprocessed.pkl` -- their fully built 47-feature tables (39 DataFrames,
    92 columns each).  Loading these reproduces their random forest exactly rather
    than re-deriving the position-context features from scratch.
  * `data/mut_matrix_alphabetical.npy` -- the global 20x20 substitution matrix
    (s~_exp) behind their null model.

Their 169 MB merged_predictions.csv and the trained model are not used.

Output: data/raw/mave_hoie/  (~250 MB, gitignored)
"""

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
URL = ("https://raw.githubusercontent.com/KULL-Centre/papers/main/"
       "2021/ML-variants-Hoie-et-al/data.zip")
OUT_DIR = ROOT / "data" / "raw" / "mave_hoie"
MEMBERS_DIR = "data/preprocessed/"
MEMBERS_EXTRA = ("data/preprocessed.pkl", "data/mut_matrix_alphabetical.npy")
N_EXPECTED = 39


def download(zip_path: Path) -> None:
    """Fetch data.zip unless it is already cached at zip_path."""
    if zip_path.exists() and zip_path.stat().st_size > 1_000_000:
        print(f"using cached archive {zip_path} "
              f"({zip_path.stat().st_size / 1e6:.0f} MB)")
        return
    print(f"downloading {URL}")
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(URL, zip_path)
    print(f"  -> {zip_path} ({zip_path.stat().st_size / 1e6:.0f} MB)")


def extract(zip_path: Path, out_dir: Path) -> list:
    """Flatten data/preprocessed/*.txt out of the archive into out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist()
                   if m.startswith(MEMBERS_DIR) and m.endswith(".txt")]
        members += [m for m in MEMBERS_EXTRA if m in zf.namelist()]
        for m in sorted(members):
            target = out_dir / Path(m).name
            with zf.open(m) as src, open(target, "wb") as dst:
                dst.write(src.read())
            written.append(target)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--zip", type=Path, default=None,
                    help="path to a pre-downloaded data.zip (default: cache "
                         "next to the output dir)")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args(argv)

    zip_path = args.zip or (args.out.parent / "mave_hoie_data.zip")
    download(zip_path)
    written = extract(zip_path, args.out)

    tables = [p for p in written if p.suffix == ".txt"]
    total = sum(p.stat().st_size for p in written)
    print(f"extracted {len(tables)} merged PRISM tables + "
          f"{len(written) - len(tables)} support files -> {args.out} "
          f"({total / 1e6:.1f} MB)")
    if len(tables) != N_EXPECTED:
        print(f"WARNING: expected {N_EXPECTED} datasets, got {len(tables)}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
