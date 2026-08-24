"""
Module: run
Description: CLI for the generalization benchmark.

    # from an experiment (reads data/processed/<exp>/features_summary.parquet):
    python -m ddg.evaluation --config experiment_configs/tsuboyama_bench_fast.yaml

    # or point straight at a parquet:
    python -m ddg.evaluation --parquet path/to/features_summary.parquet --out benchmark/

Options: --model {svr,ridge,mlp}, --cluster-map <csv>, --holdouts a,b,c.
Writes tables under <out>/ and figures under <out>/figures/.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from ddg.evaluation import cluster as cluster_mod
from ddg.evaluation import plots as plots_mod
from ddg.evaluation.benchmark import run_benchmark

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _resolve_paths(args):
    """Return (parquet_path, out_dir, wt_fasta_path_or_None)."""
    if args.config:
        from ddg.config.config_loader import ProjectConfig
        config = ProjectConfig(experiment_yaml_path=args.config,
                               internal_yaml_path=args.names_config)
        proc = Path(config.exp_processed_dir)
        parquet = Path(args.parquet) if args.parquet else proc / "features_summary.parquet"
        out = Path(args.out) if args.out else proc / "benchmark"
        fasta = proc / "wt_sequences.fasta"
        return parquet, out, (fasta if fasta.exists() else None)
    if not args.parquet:
        raise SystemExit("provide --config or --parquet")
    parquet = Path(args.parquet)
    out = Path(args.out) if args.out else parquet.parent / "benchmark"
    return parquet, out, None


def main():
    ap = argparse.ArgumentParser(description="ΔΔG generalization benchmark")
    ap.add_argument("--config", help="experiment YAML (locates the parquet)")
    ap.add_argument("--parquet", help="features_summary.parquet (overrides config)")
    ap.add_argument("--out", help="output dir (default <processed>/benchmark)")
    ap.add_argument("--model", default="mlp", choices=["hgb", "svr", "ridge", "mlp"])
    ap.add_argument("--cluster-map", help="CSV protein_id,cluster for the homology holdout")
    ap.add_argument("--build-clusters", action="store_true",
                    help="build the cluster map from wt_sequences.fasta via mmseqs")
    ap.add_argument("--min-seq-id", type=float, default=0.3)
    ap.add_argument("--holdouts", help="comma list; default all feasible")
    ap.add_argument("--drop-s", action="store_true",
                    help="exclude s-derived features (s-ablation); writes to <out>_no_s")
    ap.add_argument("--names-config", default="ddg/config/internal_config.yaml")
    args = ap.parse_args()

    parquet, out, fasta = _resolve_paths(args)
    # Keep the ablation's outputs separate from the with-s baseline.
    if args.drop_s and not args.out:
        out = out.parent / f"{out.name}_no_s"
    if not parquet.exists():
        raise SystemExit(f"features table not found: {parquet}\n"
                         f"run the features step first "
                         f"(python -m ddg run <config> --step features).")
    logger.info("loading features from %s", parquet)
    df = pd.read_csv(parquet) if str(parquet).endswith(".csv") else pd.read_parquet(parquet)

    cluster_map = None
    if args.cluster_map:
        cluster_map = cluster_mod.load_cluster_map(args.cluster_map)
    elif args.build_clusters and fasta:
        cluster_map = cluster_mod.cluster_wt_sequences(
            fasta, min_seq_id=args.min_seq_id, out_csv=out / "cluster_map.csv")

    holdouts = args.holdouts.split(",") if args.holdouts else None
    results = run_benchmark(df, model_name=args.model, out_dir=out,
                            cluster_map=cluster_map, holdouts=holdouts,
                            drop_s=args.drop_s)

    figs = plots_mod.make_all(results, out / "figures")
    logger.info("done: %d holdouts, %d figures -> %s",
                len(results.summary), len(figs), out)
    print("\n=== benchmark summary ===")
    cols = ["holdout", "n", "pooled_pearson", "pearson_mean", "pearson_sd"]
    print(results.summary[[c for c in cols if c in results.summary.columns]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
