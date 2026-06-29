"""Regex fast-pass noise removal.

Ported from LiteratureTool/latex_structuring/noise_patterns.py and extended
with IDRC/consulting-report boilerplate patterns.
"""

from __future__ import annotations

import re

NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Journal metadata / boilerplate
        r"^DOI:?\s*10\.\d{4,}/\S+$",
        r"^doi:\s*10\.\d{4,}/\S+$",
        r"^ISSN[:.]?\s*\S+$",
        r"^e-ISSN[:.]?\s*\S+$",
        r"^COPYRIGHT\s*©\s*\d{4}.*$",
        r"^©\s*\d{4}.*$",
        r"^PUBLISHED\s+\d{1,2}\s+\w+\s+\d{4}$",
        r"^RECEIVED\s+\d{1,2}\s+\w+\s+\d{4}$",
        r"^ACCEPTED\s+\d{1,2}\s+\w+\s+\d{4}$",
        r"^EDITED BY\s+.+$",
        r"^REVIEWED BY\s+.+$",
        r"^\*?CORRESPONDENCE\s+.+$",
        r"^Check for updates$",
        r"^OPEN ACCESS$",
        r"^Working\s*\|\s*paper$",
        r"^This project has received funding from .+$",
        r"^Volume\s+\d+,\s*No\.\s*\d+.*\(\d+-\d+\)$",
        r"^Issue\s+\d+/\d+\s*[•·]\s*\w+\s+\d{4}$",
        # Page artifacts
        r"^www\.\S+\.\S+$",
        r"^[\w.-]+\.(?:com|org|edu|net)$",
        r"^---\s*Page\s+\d+\s*---$",
        r"^Page\s+\d+$",
        r"^p\.\s*\d+$",
        # Lone page numbers
        r"^\d{1,4}$",
        # IDRC / consulting-report boilerplate
        r"^International Development Research Centre$",
        r"^IDRC$",
        r"^Project Completion Report$",
        r"^Page\s+\d+\s+of\s+\d+$",
        r"^\d+\s*/\s*\d+$",   # e.g. "3 / 12" page markers
        r"^Confidential$",
        r"^CONFIDENTIAL$",
        r"^Draft$",
        r"^DRAFT$",
    ]
]


def is_noise_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    return any(p.match(stripped) for p in NOISE_PATTERNS)


def filter_lines(lines: list[str]) -> list[str]:
    """Remove noise lines, preserving blank lines for structural context."""
    return [line if not is_noise_line(line) else "" for line in lines]
