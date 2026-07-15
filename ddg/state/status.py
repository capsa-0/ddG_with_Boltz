"""
Module: status
Description: Build a human-readable status view of an experiment by combining the
manifest (coarse step status) with progress counted from the filesystem
(authoritative). Used by `ddg status`.
"""

from pathlib import Path

from ddg.config.config_loader import ProjectConfig
from ddg.state.manifest import Manifest
from ddg import pipeline

# (display step, manifest key it maps to)
DISPLAY_STEPS = [
    ("prepare", "prepare"),
    ("msa", "prepare"),
    ("queries", "prepare"),
    ("predict", "predict"),
    ("slim", "slim"),
    ("features", "features"),
]


def _fmt_progress(done, expected, unit) -> str:
    if expected is None:
        return f"{done} {unit}" if done else "—"
    pct = f" ({100 * done // expected}%)" if expected else ""
    return f"{done}/{expected} {unit}{pct}"


def collect(exp_cfg, names_cfg) -> dict:
    config = ProjectConfig(exp_cfg, names_cfg)
    manifest = Manifest.load(config.exp_processed_dir,
                             experiment=config.experiment_name,
                             config_path=exp_cfg)
    rows = []
    for display, mkey in DISPLAY_STEPS:
        done, expected, unit = pipeline.progress(display, config)
        status = manifest.get_status(mkey)
        # If the manifest hasn't recorded this step, infer from the filesystem so
        # a pre-existing / externally-run experiment still reads sensibly.
        if status == "pending":
            if expected and done >= expected:
                status = "done*"
            elif done > 0:
                status = "partial*"
        rows.append({
            "step": display,
            "status": status,
            "progress": _fmt_progress(done, expected, unit),
        })
    return {
        "experiment": config.experiment_name,
        "config": str(exp_cfg),
        "config_hash_ok": manifest.config_hash_matches(exp_cfg),
        "processed_dir": str(config.exp_processed_dir),
        "rows": rows,
    }


def render(info: dict) -> str:
    lines = []
    hashmark = "ok" if info["config_hash_ok"] else "CHANGED!"
    lines.append(f"Experiment: {info['experiment']}    config: {info['config']}  (hash {hashmark})")
    lines.append(f"{'step':<10} {'status':<10} progress")
    lines.append("-" * 44)
    for r in info["rows"]:
        lines.append(f"{r['step']:<10} {r['status']:<10} {r['progress']}")
    if any("*" in r["status"] for r in info["rows"]):
        lines.append("  (* inferred from files on disk; not recorded in the manifest)")
    return "\n".join(lines)
