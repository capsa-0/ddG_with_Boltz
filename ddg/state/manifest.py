"""
Module: manifest
Description: Per-experiment manifest (manifest.json) recording step status.

The manifest is coarse bookkeeping (which steps were started/finished, job ids,
errors). It is NOT the source of truth for progress — that is derived from the
files on disk (see ddg.state.status), so a killed SLURM job never leaves a stale
"running" that lies about what actually completed.
"""

import json
import time
import hashlib
from pathlib import Path

STEP_ORDER = ["prepare", "predict", "slim", "features", "train", "eval"]


def _now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def hash_config(config_path) -> str:
    p = Path(config_path)
    if not p.exists():
        return ""
    return "sha1:" + hashlib.sha1(p.read_bytes()).hexdigest()[:12]


class Manifest:
    def __init__(self, path: Path, data: dict):
        self.path = Path(path)
        self.data = data

    @classmethod
    def load(cls, exp_dir, experiment: str = "", config_path=None) -> "Manifest":
        path = Path(exp_dir) / "manifest.json"
        if path.exists():
            data = json.loads(path.read_text())
        else:
            data = {
                "experiment": experiment,
                "config_hash": hash_config(config_path) if config_path else "",
                "created": _now(),
                "updated": _now(),
                "steps": {},
            }
        # keep hash/experiment fresh if provided
        if experiment:
            data["experiment"] = experiment
        if config_path:
            data.setdefault("config_hash", hash_config(config_path))
        return cls(path, data)

    def config_hash_matches(self, config_path) -> bool:
        stored = self.data.get("config_hash", "")
        return (not stored) or stored == hash_config(config_path)

    def set_status(self, step: str, status: str, **fields) -> None:
        entry = self.data["steps"].get(step, {})
        entry["status"] = status
        if status == "running":
            entry["started"] = _now()
            entry.pop("error", None)
        if status in ("done", "failed"):
            entry["ended"] = _now()
        entry.update(fields)
        self.data["steps"][step] = entry

    def get_status(self, step: str) -> str:
        return self.data["steps"].get(step, {}).get("status", "pending")

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data["updated"] = _now()
        self.path.write_text(json.dumps(self.data, indent=2))
