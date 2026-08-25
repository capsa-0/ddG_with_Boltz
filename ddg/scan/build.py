"""
Module: build
Description: Turn a protein sequence into a runnable full-scan experiment.

Writes two files:
  - ``data/raw/<experiment>.csv``       — the L*19 mutation table (minimal adapter)
  - ``experiment_configs/<experiment>.yaml`` — the experiment config for it

After that the scan is an ordinary pipeline run; nothing downstream knows it is a
scan:

    ./slurm/submit_scan.sh experiment_configs/scan_<name>.yaml 128 2

``--first-residue`` records the number the first residue of the given sequence
carries in the reference numbering (e.g. 32 for a mature chain that starts at
residue 32 of its precursor). The pipeline itself always works 1-based over the
sequence it was given — ``ddg.datasets.prepare`` validates ``sequence[pos-1] ==
wt_aa`` — so the offset is stored in the config and applied only when the scan is
*reported*, by ddg.scan.predict.

The generated config differs from a training config in three deliberate ways:
  - ``head.mode: inference`` — the minimal adapter then expects no ``ddg`` column.
  - ``feature.blocks: [zdiag, zpool, wtz, mtz]`` — emits the concat (``wtz``/``mtz``)
    representation the scan predictor needs alongside the historical raw-Δz pair.
  - ``slim.keep_s: false`` — s is unused by the concat model and costs disk.
Everything else (MSA strategy, Boltz flags) mirrors the configs that produced the
training corpora, so the scan's features are directly comparable to them.
"""

import argparse
import logging
from pathlib import Path

import yaml

from ddg.scan.mutations import (all_point_mutations, clean_sequence,
                                parse_positions, positions_with_residue,
                                read_fasta_sequence)

logger = logging.getLogger(__name__)

# Boltz / MSA settings copied from the configs behind the training corpora
# (experiment_configs/s669.yaml, fireprot_*.yaml, tsuboyama_bench_fast.yaml). A scan
# must see the model in the same configuration the regressor was trained under,
# otherwise its features are not on the same scale.
BOLTZ_FLAGS = {
    "accelerator": "gpu",
    "model": "boltz2",
    "recycling_steps": 3,
    "write_full_pae": False,
    "write_full_pde": False,
    "write_embeddings": True,
    "skip_run_structure": True,
    "no_kernels": True,
}
FEATURE_BLOCKS = ["zdiag", "zpool", "wtz", "mtz"]


def scan_config(experiment_name: str, raw_data_path: str,
                first_residue: int = 1, positions=None,
               experiment: str | None = None) -> dict:
    """Build the experiment-config dict for a scan."""
    scan_block = {"first_residue": int(first_residue)}
    if positions is not None:
        # Record the scanned sites (in reported numbering) so the config alone says
        # what this run covers -- a partial scan must never be mistaken for a full one.
        scan_block["n_positions"] = len(positions)
        scan_block["positions"] = [int(p) + int(first_residue) - 1 for p in positions]
    return {
        "scan": scan_block,
        "head": {
            "mode": "inference",
            "experiment_name": experiment_name,
        },
        "data_processing": {
            "overwrite": False,
            "raw_data_path": raw_data_path,
            "dataset_type": "minimal",
            "msa_strategy": "mutate_wt_msa",
            "msa_mutation_strategy": "mutate_first_row",
            "max_msa_sequences": 1000,
        },
        "feature_extraction": {
            "process_one_by_one": False,
            "boltz_flags": dict(BOLTZ_FLAGS),
        },
        "slim": {"keep_s": False, "delete_raw": True},
        "feature": {"source": "auto", "blocks": list(FEATURE_BLOCKS)},
    }


def build_scan(sequence: str, name: str, raw_dir="data/raw",
               config_dir="experiment_configs", overwrite: bool = False,
               first_residue: int = 1, positions=None,
               experiment: str | None = None) -> dict:
    """
    Write the mutation CSV and experiment YAML for a full scan of ``sequence``.

    Args:
        sequence: the wild-type protein sequence (20 standard AAs only).
        name: protein identifier; becomes ``wt_id`` and names the experiment
              ``scan_<name>``.
        overwrite: allow replacing existing CSV / YAML files.
        first_residue: number of the sequence's first residue in the reference
              numbering used to *report* results (the CSV stays 1-based).
        positions: 1-based sequence indices to scan (all 19 substitutions each).
              None scans every position.
        experiment: experiment name; defaults to ``scan_<name>``. Override it to run
              several scans of the SAME protein (a full one and a targeted subset,
              say) in separate processed dirs while keeping ``wt_id`` -- and therefore
              the MSA / query / slim-store keys -- identical, so MSAs and already
              slimmed structures can be shared between them.

    Returns a dict describing what was written.
    """
    seq = clean_sequence(sequence)
    experiment = experiment or f"scan_{name}"
    csv_path = Path(raw_dir) / f"{experiment}.csv"
    cfg_path = Path(config_dir) / f"{experiment}.yaml"
    for path in (csv_path, cfg_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")

    mutations = all_point_mutations(seq, name, positions=positions)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    mutations.to_csv(csv_path, index=False)

    config = scan_config(experiment, str(csv_path), first_residue=first_residue,
                         positions=positions)
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as fh:
        scope = ("Full single-point-mutation scan" if positions is None
                 else f"PARTIAL single-point-mutation scan ({len(positions)} of "
                      f"{len(seq)} positions)")
        fh.write(f"# {scope} of {name} ({len(seq)} aa).\n")
        fh.write(f"# {len(mutations)} mutants + 1 wild-type = "
                 f"{len(mutations) + 1} Boltz structures.\n")
        fh.write(f"# Mutation strings in the CSV are 1-based over the sequence; results\n"
                 f"# are reported with the first residue numbered {first_residue}.\n")
        fh.write("# Generated by `python -m ddg.scan build` — see ddg/scan/build.py.\n")
        yaml.safe_dump(config, fh, sort_keys=False, default_flow_style=False, indent=2)

    logger.info("scan '%s': wrote %s (%d mutations) and %s",
                experiment, csv_path, len(mutations), cfg_path)
    return {
        "experiment": experiment,
        "wt_id": name,
        "length": len(seq),
        "first_residue": int(first_residue),
        "last_residue": int(first_residue) + len(seq) - 1,
        "n_mutations": len(mutations),
        "n_positions": len(positions) if positions is not None else len(seq),
        "n_structures": len(mutations) + 1,
        "csv": str(csv_path),
        "config": str(cfg_path),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="ddg.scan build",
        description="Generate the mutation table + experiment config for a full "
                    "single-point-mutation scan of one protein")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--sequence", help="the wild-type sequence (one-letter codes)")
    src.add_argument("--fasta", help="FASTA file; its first record is used")
    ap.add_argument("--name", help="protein id (default: the FASTA header id)")
    ap.add_argument("--experiment",
                    help="experiment name (default scan_<name>); set it to run "
                         "another scan of the same protein without colliding, while "
                         "keeping wt_id -- and the MSA/slim keys -- shared")
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--config-dir", default="experiment_configs")
    ap.add_argument("--first-residue", type=int, default=1,
                    help="number of the sequence's first residue in the numbering "
                         "results should be reported in (e.g. 32 for a mature chain "
                         "starting at residue 32 of its precursor); default 1")
    ap.add_argument("--positions",
                    help="restrict the scan to these positions (all 19 substitutions "
                         "each), in the --first-residue numbering; comma list with "
                         "ranges, e.g. '80,137,169-175'")
    ap.add_argument("--wt-residues",
                    help="also scan every position whose wild-type residue is one of "
                         "these (e.g. 'G' for all glycines)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing scan CSV / config")
    args = ap.parse_args(argv)

    if args.fasta:
        header, sequence = read_fasta_sequence(args.fasta)
        name = args.name or header
    else:
        sequence, name = args.sequence, args.name
        if not name:
            ap.error("--name is required with --sequence")

    sites = None
    if args.positions or args.wt_residues:
        seq = clean_sequence(sequence)
        chosen: set[int] = set()
        if args.positions:
            chosen.update(parse_positions(args.positions, args.first_residue, len(seq)))
        if args.wt_residues:
            chosen.update(positions_with_residue(seq, args.wt_residues))
        sites = sorted(chosen)
        if not sites:
            ap.error("--positions/--wt-residues selected no positions")

    info = build_scan(sequence, name, raw_dir=args.raw_dir,
                      config_dir=args.config_dir, overwrite=args.overwrite,
                      first_residue=args.first_residue, positions=sites,
                      experiment=args.experiment)
    print(f"experiment : {info['experiment']}")
    print(f"wt_id      : {info['wt_id']}  (structure keys / MSA filenames)")
    print(f"protein    : {info['wt_id']}  ({info['length']} aa, "
          f"reported as residues {info['first_residue']}-{info['last_residue']})")
    print(f"positions  : {info['n_positions']} of {info['length']}")
    print(f"mutations  : {info['n_mutations']}  "
          f"({info['n_structures']} Boltz structures incl. the wild type)")
    print(f"csv        : {info['csv']}")
    print(f"config     : {info['config']}")
    print(f"\nNext (on the cluster):\n"
          f"  ./slurm/submit_scan.sh {info['config']} 128 2")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    raise SystemExit(main())
