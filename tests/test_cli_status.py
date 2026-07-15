"""
Test the status view: it must derive per-step progress from files on disk and
reflect manifest status. Uses a temp processed dir via a temp internal config.

Run:  PYTHONPATH=. python tests/test_cli_status.py   (or via pytest)
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ddg.state import status as status_mod
from ddg.state.manifest import Manifest


def _write_yaml(path, obj):
    Path(path).write_text(yaml.safe_dump(obj))


def test_status_reflects_disk_and_manifest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        processed = tmp / "processed"
        exp_name = "exp1"
        exp_dir = processed / exp_name
        exp_dir.mkdir(parents=True)

        # internal config pointing processed_dir at the temp dir
        internal = tmp / "internal.yaml"
        _write_yaml(internal, {
            "directories": {
                "raw_data_dir": str(tmp / "raw"),
                "processed_dir": str(processed),
                "msa_output_dir": "msas",
                "queries_dir": "queries",
                "raw_features_dir": "boltz_raw_output",
            },
            "filenames": {
                "multifasta": "wt_sequences.fasta",
                "mutations_df": "mutations.csv",
                "metadata_df": "metadata.csv",
            },
        })

        # experiment config
        raw_csv = tmp / "raw.csv"
        raw_csv.parent.mkdir(exist_ok=True)
        raw_csv.write_text("x\n")
        exp_cfg = tmp / "exp.yaml"
        _write_yaml(exp_cfg, {
            "head": {"mode": "train", "experiment_name": exp_name},
            "data_processing": {
                "overwrite": False, "raw_data_path": str(raw_csv),
                "dataset_type": "dms", "msa_strategy": "mutate_wt_msa",
                "msa_mutation_strategy": "mutate_first_row", "max_msa_sequences": 100,
            },
            "feature_extraction": {"process_one_by_one": False, "boltz_flags": {}},
            "training": {},
        })

        # Fake a prepared experiment: mutations.csv with 2 mutants of 1 WT
        pd.DataFrame([
            {"wt_id": "p1", "mutation": "D1Q", "sequence_wt": "DEV", "ddg": 0.1,
             "wt_key": "p1", "sample_key": "p1_D1Q", "position": 1},
            {"wt_id": "p1", "mutation": "E2A", "sequence_wt": "DEV", "ddg": 0.2,
             "wt_key": "p1", "sample_key": "p1_E2A", "position": 2},
        ]).to_csv(exp_dir / "mutations.csv", index=False)

        # 1 of the 3 expected predictions present -> predict should be "partial"
        preds = exp_dir / "boltz_raw_output" / "predictions" / "p1"
        preds.mkdir(parents=True)
        np.savez(preds / "embeddings_p1.npz", s=np.zeros((3, 2), dtype=np.float32))

        info = status_mod.collect(str(exp_cfg), str(internal))
        by_step = {r["step"]: r for r in info["rows"]}

        assert by_step["prepare"]["progress"].startswith("2/2")
        assert by_step["prepare"]["status"] == "done*"          # inferred from disk
        assert by_step["predict"]["progress"].startswith("1/3")  # 1 WT + 2 mut expected
        assert by_step["predict"]["status"] == "partial*"

        # Now record a manifest status and confirm it takes precedence
        m = Manifest.load(exp_dir, experiment=exp_name, config_path=str(exp_cfg))
        m.set_status("predict", "running", slurm_job="123")
        m.save()
        info2 = status_mod.collect(str(exp_cfg), str(internal))
        assert {r["step"]: r for r in info2["rows"]}["predict"]["status"] == "running"

        print(status_mod.render(info))
        print("OK: status derives progress from disk and respects the manifest")


if __name__ == "__main__":
    test_status_reflects_disk_and_manifest()
