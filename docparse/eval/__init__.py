"""docparse eval harness package.

Run a corpus through a MATRIX of (provider × prompt-variant) configs and score
each run on structure quality, gold-token fidelity, cost, and latency.
"""

from __future__ import annotations

from .prompts import (
    PROMPT_VARIANTS,
    get_variant,
    available_variants,
)
from .run import run_case, run_matrix
from .metrics import evaluate
from .report import write_report

__all__ = [
    "PROMPT_VARIANTS",
    "get_variant",
    "available_variants",
    "run_case",
    "run_matrix",
    "evaluate",
    "write_report",
]
