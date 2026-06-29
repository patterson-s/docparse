"""Markdown → passthrough read."""

from __future__ import annotations

from pathlib import Path


def md_to_markdown(path: str | Path) -> str:
    # utf-8-sig strips the BOM that Windows tools sometimes write
    return Path(path).read_text(encoding="utf-8-sig")
