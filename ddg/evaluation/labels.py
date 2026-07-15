"""
Module: labels
Description: Derive all holdout label columns from a features table.

The only inputs needed are the columns already present in
``features_summary.parquet``: ``wt_id`` (== the source protein_id), ``mutation``
(e.g. "P8A") and ``ddg``. Everything else — protein group, natural/designed flag,
source/target residue, substitution, and chemistry category — is computed here so
the benchmark is fully self-contained from the parquet. Cluster labels are the one
exception (they need an external map; see ddg.evaluation.cluster).
"""

import re

import pandas as pd

# ----- Chemistry groups (kept identical to ddg_datasets/build_benchmark_corpus.py) -----
HYDROPHOBIC = set("AVLIMFWY")
POLAR = set("STNQ")
POS = set("KRH")
NEG = set("DE")

# Metadata / target columns that are never features.
META_COLS = ("mut_id", "wt_id", "sample_id", "mutation", "ddg")
# Label columns this module adds (also excluded from the feature matrix).
LABEL_COLS = (
    "protein", "is_natural", "class_natural", "position",
    "wt_aa", "mut_aa", "substitution", "chem_category", "cluster",
)

_NATURAL_RE = re.compile(r"^[0-9][A-Za-z0-9]{3}\.pdb")


def chem_category(wt: str, mut: str) -> str:
    """Classify a WT->MUT substitution into one physicochemical bucket."""
    if mut == "P":
        return "X_to_P"
    if wt == "P":
        return "P_to_X"
    if mut == "G":
        return "X_to_G"
    if wt == "G":
        return "G_to_X"
    if mut == "C":
        return "X_to_C"
    if wt == "C":
        return "C_to_X"
    if wt in HYDROPHOBIC and mut in POLAR:
        return "hydrophobic_to_polar"
    if wt in POLAR and mut in HYDROPHOBIC:
        return "polar_to_hydrophobic"
    if wt in POS and mut in NEG:
        return "positive_to_negative"
    if wt in NEG and mut in POS:
        return "negative_to_positive"
    if wt not in (POS | NEG) and mut in (POS | NEG):
        return "neutral_to_charged"
    if wt in (POS | NEG) and mut not in (POS | NEG):
        return "charged_to_neutral"
    return "other"


def _parse_mutation(mut: str):
    """'P8A' -> ('P', 8, 'A'); position is 1-based as written."""
    m = re.match(r"^([A-Za-z])(\d+)([A-Za-z])$", str(mut))
    if not m:
        return None, None, None
    return m.group(1).upper(), int(m.group(2)), m.group(3).upper()


def add_label_columns(df: pd.DataFrame, cluster_map: dict | None = None) -> pd.DataFrame:
    """
    Return a copy of ``df`` with holdout label columns attached.

    Args:
        df: features table; must contain ``wt_id`` and ``mutation``.
        cluster_map: optional {protein_id -> cluster_id} for the cluster holdout.
    """
    for req in ("wt_id", "mutation"):
        if req not in df.columns:
            raise ValueError(f"features table is missing required column '{req}'")

    out = df.copy()
    parsed = out["mutation"].map(_parse_mutation)
    out["wt_aa"] = parsed.map(lambda t: t[0])
    out["position"] = parsed.map(lambda t: t[1])
    out["mut_aa"] = parsed.map(lambda t: t[2])
    out["substitution"] = out["wt_aa"].fillna("?") + "->" + out["mut_aa"].fillna("?")
    out["chem_category"] = [
        chem_category(w, m) if (w and m) else "other"
        for w, m in zip(out["wt_aa"], out["mut_aa"])
    ]
    out["protein"] = out["wt_id"]
    out["is_natural"] = out["wt_id"].astype(str).str.match(_NATURAL_RE).astype(int)
    out["class_natural"] = out["is_natural"].map({1: "natural", 0: "designed"})
    if cluster_map is not None:
        out["cluster"] = out["wt_id"].map(cluster_map)
    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric feature columns: everything that is not metadata or a label."""
    drop = set(META_COLS) | set(LABEL_COLS)
    return [
        c for c in df.columns
        if c not in drop and pd.api.types.is_numeric_dtype(df[c])
    ]
