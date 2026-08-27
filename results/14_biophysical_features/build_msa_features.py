"""14 (item 3) — Conservation / PSSM / consensus features from the WT MSAs.

The target is the project's one non-artifactual deficit: results/12 found the model
calls stabilizing mutations destabilizing (bias +0.56, rho 0.27), and results/13 proved
no loss reweighting fixes that on a frozen representation -- "loss reweighting can move
predictions but cannot create discrimination that the features do not carry". So the
fix has to be a feature that carries it, and the classical biological answer is
evolutionary: a mutation *toward* the family consensus is enriched in stabilizing
mutations (consensus design), and a WT residue that is rare in its own alignment column
is a stability liability the fold tolerates for functional reasons.

results/04 already showed the MSA is worth +0.08-0.10 r to Boltz *implicitly*. This
asks whether making it explicit at the mutated column adds on top.

Alignment handling: the ColabFold a3m is aligned to the query, so dropping lowercase
insertion columns leaves columns mapping 1:1 onto WT residues. Sequences are weighted
by 80%-identity clustering (w_k = 1/|cluster_k|) so a burst of near-duplicate hits
cannot dominate a column; `neff` is the sum of those weights.

    python results/14_biophysical_features/build_msa_features.py [--exp NAME ...]

Writes ``data/processed/<exp>/features_msa.parquet`` (one row per mutation).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/media/capsa/Programas/ddG_with_Boltz")
AA = "ACDEFGHIKLMNPQRSTVWY"
AA_IDX = {a: i for i, a in enumerate(AA)}
GAP = 20                      # gap / non-standard letters share one index
ID_CUT = 0.8                  # sequence-weighting identity threshold
MAX_SEQS = 2000               # cap depth before O(N^2) weighting (see read_a3m)
SUBSAMPLE_SEED = 0
PSEUDO = 1.0                  # pseudocount on the weighted column counts
# "has a usable alignment". depth > 1 is useless as a flag: the ColabFold server
# returns the query plus at least one hit for everything, so it is constant at 1.
# De novo designed proteins in this corpus sit at depth 2 (median) against 8,823
# for natural ones, so 10 cleanly separates "no homologues" from "an alignment".
MIN_DEPTH = 10

# Robinson & Robinson background amino-acid frequencies
BG = np.array([0.0787, 0.0157, 0.0530, 0.0636, 0.0400, 0.0691, 0.0227, 0.0591,
               0.0595, 0.0961, 0.0238, 0.0425, 0.0468, 0.0393, 0.0526, 0.0684,
               0.0586, 0.0673, 0.0119, 0.0324])
RES_MSA_KEYS = ("msafreq", "msalogodds", "is_consensus", "x_cons")


def read_a3m(path: Path, max_seqs: int = MAX_SEQS) -> tuple[np.ndarray, int]:
    """Parse an a3m into an (N, L) int8 array aligned to the query.

    Lowercase characters are insertions relative to the query and are dropped, so
    column j corresponds to WT residue j. Returns an empty array if unreadable.

    Depth is capped at `max_seqs` because the sequence weighting below is O(N^2):
    the cap is applied by **random subsampling** (query always kept), not by taking
    the first N. a3m rows come back E-value-ordered, so truncating would keep only
    the closest homologues and systematically understate column entropy.
    """
    seqs, cur = [], []
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if cur:
                    seqs.append("".join(cur))
                    cur = []
            else:
                cur.append(line)
    if cur:
        seqs.append("".join(cur))
    if not seqs:
        return np.zeros((0, 0), dtype=np.int8), 0
    true_depth = len(seqs)
    if len(seqs) > max_seqs:
        rng = np.random.default_rng(SUBSAMPLE_SEED)
        pick = rng.choice(len(seqs) - 1, size=max_seqs - 1, replace=False) + 1
        seqs = [seqs[0]] + [seqs[i] for i in np.sort(pick)]

    query_len = sum(1 for c in seqs[0] if not c.islower())
    rows = []
    for s in seqs:
        kept = [c for c in s if not c.islower()]      # drop insertion columns
        if len(kept) != query_len:
            continue                                   # malformed row, skip
        rows.append([AA_IDX.get(c.upper(), GAP) for c in kept])
    msa = np.asarray(rows, dtype=np.int8) if rows else np.zeros((0, 0), np.int8)
    return msa, true_depth


def sequence_weights(msa: np.ndarray, cut: float = ID_CUT) -> np.ndarray:
    """w_k = 1 / #sequences within `cut` identity of k (chunked, memory-bounded)."""
    n, L = msa.shape
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    counts = np.zeros(n, dtype=np.int32)
    step = max(1, int(4e7 // max(n * L, 1)))          # keep each block ~40M cells
    for start in range(0, n, step):
        block = msa[start:start + step]               # (b, L)
        same = (block[:, None, :] == msa[None, :, :]).sum(axis=2)   # (b, n)
        counts[start:start + step] = (same >= cut * L).sum(axis=1)
    return (1.0 / np.maximum(counts, 1)).astype(np.float32)


def column_stats(msa: np.ndarray, w: np.ndarray, true_depth: int) -> dict:
    """Per-column weighted profile, entropy, consensus and gap fraction."""
    n, L = msa.shape
    prof = np.zeros((L, 20), dtype=np.float64)
    for a in range(20):
        prof[:, a] = ((msa == a) * w[:, None]).sum(axis=0)
    gapw = ((msa == GAP) * w[:, None]).sum(axis=0)
    total = prof.sum(axis=1) + gapw
    gapfrac = np.divide(gapw, total, out=np.zeros(L), where=total > 0)

    prof += PSEUDO * BG[None, :]                       # pseudocount, then normalise
    prof /= prof.sum(axis=1, keepdims=True)
    entropy = -(prof * np.log(prof)).sum(axis=1)
    return {
        "prof": prof,                                  # (L, 20) weighted frequencies
        "entropy": entropy,
        "maxfreq": prof.max(axis=1),
        "consensus": prof.argmax(axis=1),
        "gapfrac": gapfrac,
        "neff": float(w.sum()),
        "depth": int(true_depth),
    }


def build(exp: str) -> pd.DataFrame:
    proc = ROOT / "data/processed" / exp
    muts = pd.read_csv(proc / "mutations.csv")
    msa_dir = proc / "msas"
    bio = pd.read_parquet(proc / "features_bio.parquet",
                          columns=["wt_id", "mutation", "site_cn_z"])
    cn = {(r.wt_id, r.mutation): r.site_cn_z for r in bio.itertuples(index=False)}

    stats, missing = {}, []
    for wt_id in sorted(muts.wt_id.unique()):
        path = msa_dir / f"{wt_id}.a3m"
        if not path.exists():
            missing.append(wt_id)
            continue
        msa, true_depth = read_a3m(path)
        if msa.size == 0:
            missing.append(wt_id)
            continue
        stats[wt_id] = column_stats(msa, sequence_weights(msa), true_depth)
        if len(stats) % 50 == 0:
            print(f"  [{exp}] {len(stats)} alignments processed", flush=True)
    print(f"[{exp}] {len(stats)} alignments, {len(missing)} missing", flush=True)

    nan_res = {k: np.nan for k in RES_MSA_KEYS}
    rows = []
    for m in muts.itertuples(index=False):
        pos = int(m.position) - 1
        st = stats.get(m.wt_id)
        rec = {"wt_id": m.wt_id, "mutation": m.mutation, "ddg": m.ddg}
        if st is None or pos >= len(st["entropy"]):
            rec.update({"msa_neff": np.nan, "msa_depth": np.nan,
                        "msa_gapfrac": np.nan, "msa_entropy": np.nan,
                        "msa_maxfreq": np.nan, "msa_has_msa": 0.0})
            for side in ("wt", "mt"):
                rec.update({f"{side}_{k}": v for k, v in nan_res.items()})
            rows.append(rec)
            continue

        prof = st["prof"][pos]
        rec.update({
            "msa_neff": float(np.log1p(st["neff"])),
            "msa_depth": float(np.log1p(st["depth"])),
            "msa_gapfrac": float(st["gapfrac"][pos]),
            "msa_entropy": float(st["entropy"][pos]),
            "msa_maxfreq": float(st["maxfreq"][pos]),
            # lets the model discount the whole block where there is no family to
            # be conserved against (de novo designs); msa_neff carries the same
            # information continuously
            "msa_has_msa": float(st["depth"] >= MIN_DEPTH),
        })
        z = cn.get((m.wt_id, m.mutation), 0.0)
        for side, aa in (("wt", str(m.wt_aa)), ("mt", str(m.mut_aa))):
            i = AA_IDX.get(aa)
            if i is None:
                rec.update({f"{side}_{k}": v for k, v in nan_res.items()})
                continue
            lo = float(np.log(prof[i] / BG[i]))
            rec[f"{side}_msafreq"] = float(prof[i])
            rec[f"{side}_msalogodds"] = lo
            rec[f"{side}_is_consensus"] = float(st["consensus"][pos] == i)
            # conservation means something different in the core than on the surface
            rec[f"{side}_x_cons"] = lo * (0.0 if not np.isfinite(z) else float(z))
        rows.append(rec)

    df = pd.DataFrame(rows)
    meta = ["wt_id", "mutation", "ddg"]
    feats = [c for c in df.columns if c not in meta]
    df[feats] = df[feats].astype(np.float32)
    return df[meta + feats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["tsuboyama_bench_fast", "s669"])
    args = ap.parse_args()
    for exp in args.exp:
        df = build(exp)
        out = ROOT / "data/processed" / exp / "features_msa.parquet"
        df.to_parquet(out, index=False)
        cov = df["msa_has_msa"].mean()
        print(f"[{exp}] wrote {out}  {df.shape[0]} x {df.shape[1]}  "
              f"(alignment coverage {cov:.0%})\n", flush=True)


if __name__ == "__main__":
    main()
