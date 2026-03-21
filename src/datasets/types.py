"""
Module: MutationSample
Description: Data classes for mutation sample representation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class MutationSample:
    sample_id: str
    wt_id: str
    mutation: str
    sequence_wt: str
    ddg: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
