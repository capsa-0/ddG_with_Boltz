"""
Module: cli
Description: Command-line interface for the ddg pipeline.

Usage:
    python -m ddg run    <config.yaml> [--step STEP] [--names-config PATH]
    python -m ddg status <config.yaml> [--json]
    python -m ddg list   [--processed-dir data/processed]

Steps (in order): prepare, predict, slim, features. With no --step, `run`
executes them all in order. Status is derived from the files on disk, so it is
correct even if a job was killed.
"""

import sys
import json
import logging
import argparse
from pathlib import Path

from ddg import pipeline
from ddg.config.config_loader import ProjectConfig
from ddg.state.manifest import Manifest
from ddg.state import status as status_mod

DEFAULT_NAMES_CONFIG = "ddg/config/internal_config.yaml"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ddg")


def _run_step(step: str, exp_cfg: str, names_cfg: str) -> None:
    """Run one step and record its status in the manifest."""
    config = ProjectConfig(exp_cfg, names_cfg)
    logger.info("=== step '%s' starting ===", step)

    manifest = Manifest.load(config.exp_processed_dir,
                             experiment=config.experiment_name, config_path=exp_cfg)
    if not manifest.config_hash_matches(exp_cfg):
        logger.warning("config has changed since this experiment was created "
                       "(manifest hash differs) — results may be inconsistent")
    manifest.set_status(step, "running")
    manifest.save()

    try:
        pipeline.RUN_FUNCS[step](exp_cfg, names_cfg)
    except Exception as e:  # noqa: BLE001 - record failure then re-raise
        # reload: prepare may have recreated the dir/manifest
        manifest = Manifest.load(config.exp_processed_dir,
                                 experiment=config.experiment_name, config_path=exp_cfg)
        manifest.set_status(step, "failed", error=str(e)[:500])
        manifest.save()
        logger.error("step '%s' failed: %s", step, e)
        raise

    manifest = Manifest.load(config.exp_processed_dir,
                             experiment=config.experiment_name, config_path=exp_cfg)
    manifest.set_status(step, "done")
    manifest.save()
    logger.info("=== step '%s' done ===", step)


def cmd_run(args) -> int:
    steps = [args.step] if args.step else pipeline.RUNNABLE
    for step in steps:
        _run_step(step, args.config, args.names_config)
    return 0


def cmd_status(args) -> int:
    info = status_mod.collect(args.config, args.names_config)
    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(status_mod.render(info))
    return 0


def cmd_list(args) -> int:
    root = Path(args.processed_dir)
    if not root.exists():
        print(f"(no processed dir at {root})")
        return 0
    found = False
    for manifest_path in sorted(root.glob("*/manifest.json")):
        found = True
        data = json.loads(manifest_path.read_text())
        steps = data.get("steps", {})
        done = [s for s, v in steps.items() if v.get("status") == "done"]
        running = [s for s, v in steps.items() if v.get("status") == "running"]
        state = "running:" + ",".join(running) if running else ("done:" + ",".join(done) if done else "empty")
        print(f"{data.get('experiment', manifest_path.parent.name):<30} {state}")
    if not found:
        print(f"(no experiments under {root})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ddg", description="ddG-with-Boltz pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="run pipeline step(s)")
    pr.add_argument("config", help="experiment config YAML")
    pr.add_argument("--step", choices=pipeline.RUNNABLE, help="single step (default: all in order)")
    pr.add_argument("--names-config", default=DEFAULT_NAMES_CONFIG)
    pr.set_defaults(func=cmd_run)

    ps = sub.add_parser("status", help="show step-by-step progress")
    ps.add_argument("config", help="experiment config YAML")
    ps.add_argument("--names-config", default=DEFAULT_NAMES_CONFIG)
    ps.add_argument("--json", action="store_true")
    ps.set_defaults(func=cmd_status)

    pl = sub.add_parser("list", help="list experiments and their state")
    pl.add_argument("--processed-dir", default="data/processed")
    pl.set_defaults(func=cmd_list)

    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
