"""
ddg.evaluation — generalization benchmarking for ΔΔG regressors.

Turns a single ``features_summary.parquet`` (one row per mutation, Boltz-derived
features + ddg) into the full holdout suite described in docs/benchmark_plan.md:
random / protein / cluster / de-novo / substitution / residue / chemistry splits,
each fit with the same regressor and scored with the same metrics.

Entry point: ``python -m ddg.evaluation --config <experiment.yaml>``.
See ddg/evaluation/run.py.
"""

from ddg.evaluation.benchmark import run_benchmark  # noqa: F401
