"""Run AFToolkit's released adapters on AFToolkit's released S669 AF2 features.

This produces AFToolkit's *own* per-variant S669 predictions, which is what makes the
comparison in `headtohead.py` paired: both models are scored on identical variants
instead of on a published aggregate.

Setup (once). The assets are not in the git repo; the README's download URLs are
typo'd -- the key is `s669_pkls.zip` (not `s699_`) and the `+` in the model path must
be percent-encoded, or the object store answers 403:

    AFT=$SCRATCH/aftoolkit; mkdir -p $AFT && cd $AFT
    B=https://bioinformatics-kardymon.obs.ru-moscow-1.hc.sbercloud.ru/AFToolkit
    curl -L -o stability_task_files.zip "$B/data/stability_task_files.zip" && unzip -q -o stability_task_files.zip
    for m in svm mlp catboost; do curl -L -o trained_$m.pkl \
      "$B/models/monomer/pair%2Blddt_logits%2Bplddt/trained_${m}_concat_nomultitrain_aggmutpos_multisum.pkl"; done
    curl -L -o s669_pkls.zip "$B/data/s669_pkls.zip" && unzip -q -o s669_pkls.zip -d p \
      && mv p/mnt/nfs_protein/msindeeva/aftoolkit/s669_pkls . && rm -rf p s669_pkls.zip

    git clone --depth 1 https://github.com/AIRI-Institute/AFToolkit.git repo

The pickles need `AFToolKit.*` importable (note the capital K -- the repo directory is
spelled `AFToolkit`). Only `models/adapter.py`, `models/utils.py`,
`processing/protein_task.py` and `processing/utils.py` are needed; the package
`processing/__init__.py` pulls in torch, so this script builds a shim tree with an
empty one. Unpickling the adapters needs scikit-learn 1.4.x (they were trained on it);
1.6+ raises on the Pipeline.

    python -m venv aftenv && ./aftenv/bin/pip install "scikit-learn==1.4.2" "numpy<2" \
        scipy pandas biopandas catboost tqdm
    AFT=$AFT ./aftenv/bin/python results/16_aftoolkit_headtohead/run_aftoolkit_s669.py

Writes `aftoolkit_s669_predictions.csv`: all 669 S669 variants with AFToolkit's SVM /
MLP / CatBoost predictions, mapped onto this project's (uniprot, mutation) variant ids.
"""
import os
import pickle
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

warnings.filterwarnings("ignore")
R = Path(__file__).resolve().parent
ROOT = R.parents[1]
AFT = Path(os.environ.get("AFT", "/tmp/aftoolkit"))
ADAPTERS = ["svm", "mlp", "catboost"]
FEATS = ["pair", "lddt_logits", "plddt"]


def build_shim():
    """Importable `AFToolKit` with no torch dependency."""
    shim = AFT / "shim"
    if not (shim / "AFToolKit/processing/protein_task.py").exists():
        src = AFT / "repo/AFToolkit"
        for sub in ("models", "processing"):
            (shim / "AFToolKit" / sub).mkdir(parents=True, exist_ok=True)
        (shim / "AFToolKit/__init__.py").write_text("")
        (shim / "AFToolKit/processing/__init__.py").write_text(
            "# emptied on purpose: upstream imports OpenFoldWrapper (torch) here\n")
        for f in ("models/__init__.py", "models/adapter.py", "models/utils.py",
                  "processing/protein_task.py", "processing/utils.py"):
            shutil.copy(src / f, shim / "AFToolKit" / f)
    sys.path.insert(0, str(shim))


def run_adapters(task):
    """AFToolkit's own scoring path: pkl -> adapter input features -> base model."""
    for name in ADAPTERS:
        model = pickle.load(open(AFT / f"trained_{name}.pkl", "rb"))
        X = []
        for i in task.index:
            with open(AFT / f"s669_pkls/{task.loc[i, 'id']}.pkl", "rb") as f:
                X.append(model.protein_task_to_input_features(pickle.load(f)))
        X = np.stack(X)
        task[f"aft_{name}"] = model.base_model.predict(X)
        y = task.ddg.to_numpy(float)
        print(f"  {name:9s} dim={X.shape[1]}  rho={spearmanr(y, task[f'aft_{name}']).statistic:.3f}"
              f"  r={pearsonr(y, task[f'aft_{name}'])[0]:.3f}"
              f"  rmse={np.sqrt(np.mean((y - task[f'aft_{name}']) ** 2)):.3f}")
    return task


def map_to_our_ids(task):
    """AFToolkit indexes S669 by PDB id in PDB numbering; this project indexes it by
    UniProt accession in UniProt numbering (the DDGemb mapping), and with the opposite
    ddG sign. Neither the row order nor the residue numbers line up, so match the two
    669-row tables one-to-one on (wild-type aa, mutant aa, ddG), preferring the PDB
    entry that most of a UniProt's variants vote for. Every row must match exactly."""
    ours = pd.read_csv(ROOT / "data/raw/s669_full669.csv").reset_index(drop=True)
    ours["wt_aa"], ours["mt_aa"] = ours.mutation.str[0], ours.mutation.str[-1]
    task = task.reset_index(drop=True)
    task["wt_aa"], task["mt_aa"] = task.mut_info.str[0], task.mut_info.str[-1]
    task["ddg_ours"] = -task.ddg               # opposite sign convention
    ours["r2"], task["r2"] = ours.ddg.round(2), task.ddg_ours.round(2)

    votes = {u: {} for u in ours.uniprot.unique()}
    for o in ours.itertuples():
        for a in task[(task.wt_aa == o.wt_aa) & (task.mt_aa == o.mt_aa)
                      & (task.r2 == o.r2)].itertuples():
            votes[o.uniprot][a.pdb_id] = votes[o.uniprot].get(a.pdb_id, 0) + 1

    BIG = 1e6
    C = np.full((len(ours), len(task)), BIG)
    for i, o in enumerate(ours.itertuples()):
        ok = ((task.wt_aa.values == o.wt_aa) & (task.mt_aa.values == o.mt_aa)
              & (task.r2.values == o.r2))
        C[i, ok] = 1.0
        if votes[o.uniprot]:
            top = max(votes[o.uniprot], key=votes[o.uniprot].get)
            C[i, ok & (task.pdb_id.values == top)] = 0.0
    rows, cols = linear_sum_assignment(C)
    assert (C[rows, cols] < BIG).all(), "some S669 variants could not be matched"
    ours["aft_row"] = cols
    assert np.abs(ours.ddg.values - task.ddg_ours.values[cols]).max() < 1e-9
    bad = [r for i, r in enumerate(ours.itertuples())
           if task.pdb_id.values[cols[i]] not in votes[r.uniprot]]
    assert not bad, "an assignment fell outside its protein's candidate PDB entries"
    for c in ADAPTERS:
        ours[f"aft_{c}"] = task[f"aft_{c}"].values[cols]
    ours["aft_pdb"] = task.pdb_id.values[cols]
    ours["aft_mut_info"] = task.mut_info.values[cols]
    return ours.drop(columns=["wt_aa", "mt_aa", "r2", "aft_row"])


def main():
    build_shim()
    task = pd.read_csv(AFT / "s669_mut_idxs.csv", index_col=0)
    print(f"AFToolkit S669 task file: {len(task)} variants / {task.pdb_id.nunique()} proteins")
    task = run_adapters(task)
    out = map_to_our_ids(task)
    out.to_csv(R / "aftoolkit_s669_predictions.csv", index=False)
    print(f"wrote {R/'aftoolkit_s669_predictions.csv'} ({len(out)} variants)")


if __name__ == "__main__":
    main()
