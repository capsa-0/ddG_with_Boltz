"""14 — Build the biology-informed feature blocks (items 1 and 2).

Reads the slim embedding store and emits, per mutation, blocks that the current
pipeline does not have:

ITEM 1 — contact-weighted pooling.
  ``ddg/features/build_features.py`` pools ``z[i, :]`` **uniformly** over every
  residue of the chain, so a residue 60 A away weighs as much as a contacting one.
  Here the pooling weight is Boltz's own predicted distogram at the WT structure:

      w_ij = P(d_ij < CUT)  from softmax(pdrow_wt[i, j, :]),  |i-j| > 2, normalised

  and the same WT weights are applied to both the WT and the mutant z row, so the
  pair stays a clean concat block (the antisymmetry swap wtcw <-> mtcw is exact).

    wtcw_/mtcw_   : near shell (CUT = 8 A)
    wtfar_/mtfar_ : the complement P(d >= 8 A), renormalised -- the control block

ITEM 2 — burial + per-residue biophysics + their interaction.
    site_*        : contact numbers at 8/10/12 A, within-protein z-scored burial,
                    relative position, chain length, distance to the nearer terminus
                    (invariant under mutation reversal -- never swapped)
    wt_*/mt_*     : volume, hydropathy, transfer free energy, charge, polarity,
                    aromaticity, helix/sheet propensity, flexibility, Gly/Pro flags
    wt_x_*/mt_x_* : each residue scalar times the within-protein burial z -- the
                    "big side chain IN A BURIED SITE" term, which is the actual
                    biophysics and which a plain MLP has to discover from scratch

Distogram binning is 64 bins over 2-22 A (boundaries ``linspace(2, 22, 63)``), and
the contact-number convention is the one results/12 used, so ``site_cn10`` is
directly comparable to that study's burial tertiles.

    python results/14_biophysical_features/build_bio_features.py [--exp NAME ...]

Writes ``data/processed/<exp>/features_bio.parquet``.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ddg.storage.slim_store import SlimStore

# Repo root. Derived from this file's location so the script runs on the cluster as
# well as the workstation it was written on; DDG_ROOT overrides it if ever needed.
ROOT = Path(os.environ.get("DDG_ROOT", Path(__file__).resolve().parents[2]))
Z_DIM = 128
BIN_BOUNDS = np.linspace(2.0, 22.0, 63)   # 63 boundaries -> 64 distogram bins
SEQ_SEP = 2                               # ignore |i-j| <= 2 (trivial backbone neighbours)
NEAR_CUT = 8.0                            # first-shell cutoff for the pooling weights
CN_CUTS = (8.0, 10.0, 12.0)

# ---------------------------------------------------------------- residue scales
VOL = dict(A=88.6, R=173.4, N=114.1, D=111.1, C=108.5, Q=143.8, E=138.4, G=60.1,
           H=153.2, I=166.7, L=166.7, K=168.6, M=162.9, F=189.9, P=112.7, S=89.0,
           T=116.1, W=227.8, Y=193.6, V=140.0)
# Kyte-Doolittle hydropathy
KD = dict(A=1.8, R=-4.5, N=-3.5, D=-3.5, C=2.5, Q=-3.5, E=-3.5, G=-0.4, H=-3.2,
          I=4.5, L=3.8, K=-3.9, M=1.9, F=2.8, P=-1.6, S=-0.8, T=-0.7, W=-0.9,
          Y=-1.3, V=4.2)
# Fauchere-Pliska octanol-water transfer free energy (kcal/mol, relative to Gly)
DGT = dict(A=0.31, R=-1.01, N=-0.60, D=-0.77, C=1.54, Q=-0.22, E=-0.64, G=0.00,
           H=0.13, I=1.80, L=1.70, K=-0.99, M=1.23, F=1.79, P=0.72, S=-0.04,
           T=0.26, W=2.25, Y=0.96, V=1.22)
CHARGE = dict(D=-1.0, E=-1.0, K=1.0, R=1.0, H=0.1)
POLAR = set("STNQCYHDEKR")
AROMATIC = set("FWYH")
# Chou-Fasman propensities
P_ALPHA = dict(E=1.51, M=1.45, A=1.42, L=1.21, K=1.16, F=1.13, Q=1.11, W=1.08,
               I=1.08, V=1.06, D=1.01, H=1.00, R=0.98, T=0.83, S=0.77, C=0.70,
               Y=0.69, N=0.67, P=0.57, G=0.57)
P_BETA = dict(V=1.70, I=1.60, Y=1.47, F=1.38, W=1.37, L=1.30, C=1.19, T=1.19,
              Q=1.10, M=1.05, R=0.93, N=0.89, H=0.87, A=0.83, S=0.75, G=0.75,
              K=0.74, P=0.55, D=0.54, E=0.37)
# Vihinen average flexibility index
FLEX = dict(A=0.984, C=0.906, D=1.068, E=1.094, F=0.915, G=1.031, H=0.950,
            I=0.927, K=1.102, L=0.935, M=0.952, N=1.048, P=1.049, Q=1.037,
            R=1.008, S=1.046, T=0.997, V=0.931, W=0.904, Y=0.929)

RES_SCALES = [("vol", VOL), ("hyd", KD), ("dgtrans", DGT)]
# scalars that get multiplied by the within-protein burial z-score
INTERACT = ("vol", "hyd", "dgtrans", "charge", "is_gly", "is_pro")


def residue_scalars(aa: str) -> dict:
    """The 11 per-residue biophysical scalars (NaN for a non-standard letter)."""
    if aa not in VOL:
        return {k: np.nan for k in
                ("vol", "hyd", "dgtrans", "charge", "polar", "aromatic",
                 "helixprop", "sheetprop", "flex", "is_gly", "is_pro")}
    return {
        "vol": VOL[aa], "hyd": KD[aa], "dgtrans": DGT[aa],
        "charge": CHARGE.get(aa, 0.0),
        "polar": float(aa in POLAR), "aromatic": float(aa in AROMATIC),
        "helixprop": P_ALPHA[aa], "sheetprop": P_BETA[aa], "flex": FLEX[aa],
        "is_gly": float(aa == "G"), "is_pro": float(aa == "P"),
    }


# ---------------------------------------------------------------- distogram maths
def _softmax_rows(logits: np.ndarray) -> np.ndarray:
    """Row-wise softmax over the last (bin) axis, in float32."""
    x = logits.astype(np.float32)
    x = x - x.max(axis=-1, keepdims=True)
    np.exp(x, out=x)
    x /= x.sum(axis=-1, keepdims=True)
    return x


def _bins_below(cut: float) -> np.ndarray:
    """Boolean mask of distogram bins whose lower edge is below `cut`.

    Bin 0 is [0, 2 A); bin k (k>=1) starts at BIN_BOUNDS[k-1]. Same convention as
    results/12_error_anatomy/tsu_class_error.py, so contact numbers are comparable.
    """
    return np.concatenate([[True], BIN_BOUNDS < cut])


def contact_profile(pdrow_pos: np.ndarray, pos: int) -> dict:
    """From one distogram row: P(d<cut) per residue plus the contact numbers.

    `pdrow_pos` is (L, 64) logits for the mutated residue's row.
    """
    p = _softmax_rows(pdrow_pos)                       # (L, 64)
    L = p.shape[0]
    far_mask = np.abs(np.arange(L) - pos) > SEQ_SEP    # drop self + i+-1, i+-2
    out = {}
    for cut in CN_CUTS:
        w = p[:, _bins_below(cut)].sum(axis=-1)        # (L,) P(d_ij < cut)
        w = np.where(far_mask, w, 0.0).astype(np.float32)
        out[cut] = w
    return out


def _normalise(w: np.ndarray) -> np.ndarray:
    """Turn a non-negative weight vector into a weighted-average kernel."""
    total = float(w.sum())
    if not np.isfinite(total) or total <= 1e-6:
        n = w.size
        return np.full(n, 1.0 / n, dtype=np.float32)   # degenerate -> uniform
    return (w / total).astype(np.float32)


# ---------------------------------------------------------------- per-experiment
def build(exp: str) -> pd.DataFrame:
    proc = ROOT / "data/processed" / exp
    muts = pd.read_csv(proc / "mutations.csv")
    store = SlimStore(proc / "slim")
    print(f"[{exp}] {len(muts)} mutations / {muts.wt_id.nunique()} proteins; "
          f"{len(store.index)} structures in slim store", flush=True)

    muts = muts.sort_values("wt_key").reset_index(drop=True)
    rows, missing = [], 0
    cache_key, cache = None, None

    for n, m in enumerate(muts.itertuples(index=False), 1):
        if m.wt_key not in store or m.sample_key not in store:
            missing += 1
            continue
        if cache_key != m.wt_key:                      # one WT held at a time (RAM)
            wt = store.get(m.wt_key)
            cache_key, cache = m.wt_key, {
                "pos": [int(p) for p in wt["pos"]],
                "zrow": wt["zrow"], "pdrow": wt["pdrow"],
            }
        pos = int(m.position) - 1                      # mutation strings are 1-based
        if pos not in cache["pos"]:
            missing += 1
            continue
        wi = cache["pos"].index(pos)
        wt_row = cache["zrow"][wi].astype(np.float32)          # (L, Dz)
        mut = store.get(m.sample_key)
        mut_row = mut["zrow"][0].astype(np.float32)            # (L, Dz)
        L = wt_row.shape[0]

        prof = contact_profile(cache["pdrow"][wi], pos)
        near_raw = prof[NEAR_CUT]
        w_near = _normalise(near_raw)
        # complement: P(d >= NEAR_CUT). near_raw is already 0 exactly where the
        # |i-j| mask bites, so 1 - near_raw is the true far probability there.
        far_mask = np.abs(np.arange(L) - pos) > SEQ_SEP
        far_raw = np.where(far_mask, 1.0 - near_raw, 0.0).astype(np.float32)
        w_far = _normalise(np.maximum(far_raw, 0.0))

        rec = {"wt_id": m.wt_id, "mutation": m.mutation, "ddg": m.ddg}
        for name, w, row in (("wtcw", w_near, wt_row), ("mtcw", w_near, mut_row),
                             ("wtfar", w_far, wt_row), ("mtfar", w_far, mut_row)):
            vec = w @ row                                       # (Dz,)
            rec.update({f"{name}_{j}": float(v) for j, v in enumerate(vec)})

        # --- item 2: site block (swap-invariant) ---
        for cut in CN_CUTS:
            rec[f"site_cn{int(cut)}"] = float(prof[cut].sum())
        rec["site_relpos"] = (pos + 1) / L
        rec["site_len"] = float(L)
        rec["site_termdist"] = float(min(pos, L - 1 - pos))

        # --- item 2: residue blocks (swapped on augmentation) ---
        for side, aa in (("wt", m.wt_aa), ("mt", m.mut_aa)):
            for k, v in residue_scalars(str(aa)).items():
                rec[f"{side}_{k}"] = v
        rows.append(rec)

        if n % 2000 == 0:
            print(f"  [{exp}] {n}/{len(muts)}", flush=True)

    store.close()
    df = pd.DataFrame(rows)
    if missing:
        print(f"  [{exp}] WARNING {missing} mutations skipped "
              f"(structure or position absent from the slim store)", flush=True)

    # within-protein burial z-score: burial must be a SITE property, not a smuggled
    # protein-level one (results/11 closed protein-level terms).
    g = df.groupby("wt_id")["site_cn10"]
    sd = g.transform("std").replace(0.0, np.nan)
    df["site_cn_z"] = ((df["site_cn10"] - g.transform("mean")) / sd).fillna(0.0)

    # interaction terms: residue scalar x within-protein burial
    for side in ("wt", "mt"):
        for k in INTERACT:
            df[f"{side}_x_{k}"] = df[f"{side}_{k}"] * df["site_cn_z"]

    meta = ["wt_id", "mutation", "ddg"]
    feats = [c for c in df.columns if c not in meta]
    df[feats] = df[feats].astype(np.float32)
    return df[meta + feats]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+",
                    default=["tsuboyama_bench_fast", "s669"])
    args = ap.parse_args()
    for exp in args.exp:
        df = build(exp)
        out = ROOT / "data/processed" / exp / "features_bio.parquet"
        df.to_parquet(out, index=False)
        print(f"[{exp}] wrote {out}  {df.shape[0]} x {df.shape[1]}\n", flush=True)


if __name__ == "__main__":
    main()
