"""results/17 Phase 0 — validate a backbone's embeddings against the NPZ contract.

Every arm must write, per structure:

    <raw_features_dir>/predictions/<key>/embeddings_<key>.npz
        s           (L x Ds)        optional, only read when slim.keep_s
        z           (L x L x Dz)    required -- this is where zdiag comes from
        pdistogram  (L x L x bins)  optional

`ddg.storage.slim` tolerates a leading recycling-step dimension and squeezes a
batch dimension, so this check applies the same collapse before asserting shapes.

Two classes of check:

  **contract**  shapes agree, L is consistent across tensors, z is square in its
                first two axes, nothing is NaN/Inf, and Dz is reported (all six
                results/17 arms should say 128).

  **sanity**    for a (WT, mutant) pair from mutations.csv: z[i,i] must *move* at
                the mutated position i, and must move much less at residues far
                from i. A backbone whose embeddings are position-blind would pass
                the contract check and produce a silently meaningless zdiag.

This reads the **raw** prediction NPZs, not the slim store: the locality check needs
the whole `z[j,j]` diagonal, while slim keeps only `z[pos, :, :]`. So the config
under test must set `slim.delete_raw: false` -- `predict_array.sbatch` self-slims
each shard and deletes its raw immediately otherwise.

Compute must not run on the login node (see CLAUDE.md), so submit it:

    srun --partition=cpu --time=00:20:00 --mem=8G \
        python results/17_backbone_transfer/validate_contract.py \
        --config experiment_configs/ssym_esmfold.yaml --pairs 20
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ddg.config.config_loader import ProjectConfig  # noqa: E402


def _collapse(a, ndim):
    """Drop a leading recycling-step / batch axis, as ddg.storage.slim does."""
    a = np.squeeze(a)
    return a[-1] if a.ndim == ndim + 1 else a


def load(npz_path):
    with np.load(npz_path) as d:
        out = {"z": _collapse(d["z"], 3)}
        if "s" in d:
            out["s"] = _collapse(d["s"], 2)
        if "pdistogram" in d:
            out["pdistogram"] = _collapse(d["pdistogram"], 3)
    return out


def check_contract(key, arrays):
    """Return (ok, [messages]) for one structure."""
    msgs, ok = [], True
    z = arrays["z"]
    if z.ndim != 3:
        return False, [f"{key}: z has ndim {z.ndim}, expected 3 (L,L,Dz)"]
    L, L2, dz = z.shape
    if L != L2:
        ok = False
        msgs.append(f"{key}: z is {L}x{L2}, not square")
    for name, arr, want_ndim in (("s", arrays.get("s"), 2),
                                 ("pdistogram", arrays.get("pdistogram"), 3)):
        if arr is None:
            continue
        if arr.ndim != want_ndim:
            ok = False
            msgs.append(f"{key}: {name} ndim {arr.ndim}, expected {want_ndim}")
        elif arr.shape[0] != L:
            ok = False
            msgs.append(f"{key}: {name} length {arr.shape[0]} != z length {L}")
    for name, arr in arrays.items():
        bad = int((~np.isfinite(arr)).sum())
        if bad:
            ok = False
            msgs.append(f"{key}: {name} has {bad} non-finite values")
    return ok, msgs


def check_sanity(wt, mut, pos0, key):
    """z[i,i] must move at the mutated position and move far less away from it."""
    zw, zm = wt["z"], mut["z"]
    if zw.shape != zm.shape:
        return False, [f"{key}: WT z {zw.shape} != mutant z {zm.shape}"]
    L = zw.shape[0]
    if not 0 <= pos0 < L:
        return False, [f"{key}: position {pos0} outside chain length {L}"]

    diag_w = np.einsum("iid->id", np.ascontiguousarray(zw)).astype(np.float64)
    diag_m = np.einsum("iid->id", np.ascontiguousarray(zm)).astype(np.float64)
    delta = np.linalg.norm(diag_m - diag_w, axis=1)      # (L,) per-residue shift

    at = float(delta[pos0])
    far = np.delete(delta, slice(max(0, pos0 - 5), min(L, pos0 + 6)))
    far_med = float(np.median(far)) if far.size else float("nan")

    ok, msgs = True, []
    if not at > 0:
        ok = False
        msgs.append(f"{key}: |Δz[i,i]| is 0 at the mutated position — the backbone "
                    f"is not seeing the substitution")
    if far.size and far_med > 0 and at < far_med:
        ok = False
        msgs.append(f"{key}: |Δz[i,i]| at the mutation ({at:.4g}) is below the "
                    f"far-residue median ({far_med:.4g}) — signal is not local")
    # Always report the numbers: Phase 0 is a judgement call about how localised
    # a backbone's response is, not just a pass/fail. far_med == 0 means the
    # diagonal moved *only* at the mutated position.
    if at == 0 and far_med == 0:
        ratio = "n/a (nothing moved anywhere)"
    elif far_med == 0:
        ratio = "inf (far field unchanged)"
    else:
        ratio = f"{at / far_med:.1f}x"
    msgs.append(f"{key}: Δ at mutation {at:.4g}, far median {far_med:.4g}, "
                f"ratio {ratio}")
    return ok, msgs


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--names-config", default="ddg/config/internal_config.yaml")
    ap.add_argument("--pairs", type=int, default=5,
                    help="how many (WT, mutant) pairs to sanity-check")
    ap.add_argument("--limit", type=int, default=50,
                    help="how many structures to contract-check")
    args = ap.parse_args()

    config = ProjectConfig(args.config, args.names_config)
    preds = Path(config.raw_features_dir) / "predictions"
    if not preds.exists():
        sys.exit(f"no predictions under {preds} — run the predict step first, and "
                 f"check the config has slim.delete_raw: false (the predict array "
                 f"self-slims and deletes raw NPZs, which this check needs)")

    found = {p.name: next(iter(p.glob("embeddings_*.npz")), None)
             for p in sorted(preds.iterdir()) if p.is_dir()}
    found = {k: v for k, v in found.items() if v is not None}
    if not found:
        sys.exit(f"no embeddings_*.npz under {preds}")
    print(f"backbone: {config.backbone}   structures on disk: {len(found)}\n")

    failures = []
    widths, lengths = set(), {}
    for key in list(found)[:args.limit]:
        arrays = load(found[key])
        ok, msgs = check_contract(key, arrays)
        if arrays["z"].ndim == 3:
            widths.add(arrays["z"].shape[2])
            lengths[key] = arrays["z"].shape[0]
        if not ok:
            failures.extend(msgs)
        for m in msgs:
            print("  " + m)
    print(f"contract: checked {min(len(found), args.limit)} structures, "
          f"z widths seen: {sorted(widths)}")
    if len(widths) > 1:
        failures.append(f"inconsistent z widths across structures: {sorted(widths)}")
    if widths and 128 not in widths:
        print(f"  NOTE: Dz={sorted(widths)} is not 128 — zdiag will not be "
              f"dimension-matched to the Boltz-2 arm. build_features derives the "
              f"width, so this is legal, but the comparison is no longer "
              f"like-for-like.")

    muts = pd.read_csv(config.mutations_df_path)
    n = 0
    print()
    for row in muts.itertuples(index=False):
        if n >= args.pairs:
            break
        wt_f, mut_f = found.get(row.wt_key), found.get(row.sample_key)
        if not (wt_f and mut_f):
            continue
        ok, msgs = check_sanity(load(wt_f), load(mut_f), int(row.position) - 1,
                                f"{row.wt_key}/{row.mutation}")
        for m in filter(None, msgs):
            print("  " + m)
        if not ok:
            failures.append(msgs[0])
        n += 1
    print(f"sanity: checked {n} (WT, mutant) pairs")

    print()
    if failures:
        print(f"FAILED — {len(failures)} problem(s)")
        sys.exit(1)
    print("PASSED — embeddings satisfy the contract and the mutation is local")


if __name__ == "__main__":
    main()
