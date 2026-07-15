"""
Module: ids
Description: Single source of truth for turning biological identifiers into
filesystem-safe keys.

The pipeline uses these keys consistently for MSA filenames, Boltz query
filenames, Boltz prediction folder names, and embedding lookups. Keeping the
sanitization in one place prevents the class of bug where an id is sanitized when
writing a file but looked up un-sanitized later (which silently breaks any id
containing characters like '|' or '/', e.g. Tsuboyama ids such as
'EA|run2_0325_0005.pdb').
"""

import re

# Characters that are NOT word-chars, dash, or dot are replaced with '_'.
# This matches how Boltz query filenames (and therefore prediction folders) are
# named, so lookups line up.
_UNSAFE = re.compile(r"[^\w\-.]")


def sanitize_id(raw: str) -> str:
    """Return a filesystem-safe version of an arbitrary identifier."""
    return _UNSAFE.sub("_", str(raw))


def wt_key(wt_id: str) -> str:
    """Canonical key for a wild-type structure."""
    return sanitize_id(wt_id)


def mutant_key(wt_id: str, mutation: str) -> str:
    """Canonical key for a mutant structure ('<wt>_<mutation>', sanitized)."""
    return sanitize_id(f"{wt_id}_{mutation}")
