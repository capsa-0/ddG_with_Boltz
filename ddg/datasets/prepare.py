"""
Module: prepare
Description: Validate and canonicalize a mutations dataframe before the pipeline
runs Boltz on it.

Responsibilities:
- Parse the mutation string ('<WT><1-based-pos><MUT>', e.g. 'P8A').
- Validate that the wild-type amino acid matches the sequence at that position
  (catches offset/isoform bugs and wrong data — bug 1.3).
- Restrict to the 20 standard amino acids.
- Attach canonical, filesystem-safe keys (wt_key / sample_key) so downstream
  naming and embedding lookups are consistent (bug 1.1).
- Return a per-run report of exactly how many rows were dropped and why, so the
  cleaning is auditable.
"""

import re
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional

import pandas as pd

from ddg.datasets.ids import wt_key as _wt_key, mutant_key as _mut_key

logger = logging.getLogger(__name__)

STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")
_MUTATION_RE = re.compile(r"^([A-Za-z])(\d+)([A-Za-z])$")


def parse_mutation(mutation: str):
    """
    Parse '<WT><pos><MUT>' into (wt_aa, pos_1based, mut_aa), upper-cased.

    Returns None if the string does not match the expected format.
    """
    if not isinstance(mutation, str):
        return None
    m = _MUTATION_RE.match(mutation.strip())
    if not m:
        return None
    wt_aa, pos, mut_aa = m.group(1).upper(), int(m.group(2)), m.group(3).upper()
    return wt_aa, pos, mut_aa


@dataclass
class PrepareReport:
    """Auditable summary of the cleaning step."""
    input_rows: int = 0
    output_rows: int = 0
    n_proteins: int = 0
    dropped_bad_format: int = 0
    dropped_nonstandard_aa: int = 0
    dropped_out_of_range: int = 0
    dropped_wt_mismatch: int = 0
    examples_wt_mismatch: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def prepare_mutations_frame(
    mutations_df: pd.DataFrame,
    strict: bool = False,
) -> tuple[pd.DataFrame, PrepareReport]:
    """
    Validate and canonicalize a mutations dataframe.

    Expects columns: sample_id, wt_id, mutation, sequence_wt, ddg.
    Adds columns: wt_aa, position (1-based), mut_aa, wt_key, sample_key.

    Args:
        mutations_df: raw frame from a dataset adapter.
        strict: if True, raise on any dropped row instead of just reporting.

    Returns:
        (clean_df, report)
    """
    report = PrepareReport(input_rows=len(mutations_df))
    clean_rows = []

    for row in mutations_df.itertuples(index=False):
        mutation = getattr(row, "mutation")
        sequence = getattr(row, "sequence_wt")
        wt_id = getattr(row, "wt_id")

        parsed = parse_mutation(mutation)
        if parsed is None:
            report.dropped_bad_format += 1
            continue
        wt_aa, pos, mut_aa = parsed

        if wt_aa not in STANDARD_AA or mut_aa not in STANDARD_AA:
            report.dropped_nonstandard_aa += 1
            continue

        if not isinstance(sequence, str) or pos < 1 or pos > len(sequence):
            report.dropped_out_of_range += 1
            continue

        if sequence[pos - 1] != wt_aa:
            report.dropped_wt_mismatch += 1
            if len(report.examples_wt_mismatch) < 10:
                report.examples_wt_mismatch.append(
                    {"wt_id": str(wt_id), "mutation": str(mutation),
                     "expected": wt_aa, "found": sequence[pos - 1], "position": pos}
                )
            continue

        record = row._asdict()
        record["wt_aa"] = wt_aa
        record["position"] = pos
        record["mut_aa"] = mut_aa
        record["wt_key"] = _wt_key(wt_id)
        record["sample_key"] = _mut_key(wt_id, mutation)
        clean_rows.append(record)

    clean_df = pd.DataFrame(clean_rows)
    report.output_rows = len(clean_df)
    report.n_proteins = int(clean_df["wt_id"].nunique()) if len(clean_df) else 0

    dropped = report.input_rows - report.output_rows
    if dropped:
        logger.warning(
            "prepare: dropped %d/%d rows "
            "(bad_format=%d, nonstandard_aa=%d, out_of_range=%d, wt_mismatch=%d)",
            dropped, report.input_rows,
            report.dropped_bad_format, report.dropped_nonstandard_aa,
            report.dropped_out_of_range, report.dropped_wt_mismatch,
        )
        if report.dropped_wt_mismatch:
            logger.warning(
                "prepare: %d rows had a WT/sequence mismatch — this often means a "
                "position-offset or wrong-isoform bug, not just noise. Examples: %s",
                report.dropped_wt_mismatch, report.examples_wt_mismatch[:3],
            )
    if strict and dropped:
        raise ValueError(f"prepare: {dropped} invalid rows and strict=True")

    return clean_df, report
