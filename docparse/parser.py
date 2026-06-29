"""Main orchestration: read → noise filter → metadata → heading detect → chunk."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import readers
from . import noise_filter
from . import metadata_extractor
from . import heading_detector
from . import chunker
from .models import ParsedDoc


def parse(path: str | Path, model: str = "mistral-medium-latest", api_key: str = "") -> ParsedDoc:
    """Parse a PDF, DOCX, or MD file into a structured ParsedDoc."""
    path = Path(path)
    fmt = readers.detect_format(path)

    raw_markdown = readers.read(path, api_key=api_key)
    lines = raw_markdown.splitlines()
    clean_lines = noise_filter.filter_lines(lines)
    clean_markdown = "\n".join(clean_lines)

    metadata = metadata_extractor.extract(clean_markdown, model=model, api_key=api_key)
    sections = heading_detector.detect(clean_lines, model=model, api_key=api_key)

    doc_id = _to_slug(path.stem)
    chunks = chunker.build_chunks(doc_id, sections, clean_markdown)

    return ParsedDoc(
        document_id=doc_id,
        filename=path.name,
        source_format=fmt,
        metadata=metadata,
        sections=sections,
        chunks=chunks,
        raw_markdown=raw_markdown,
        parsed_at=datetime.now(timezone.utc).isoformat(),
    )


def save(doc: ParsedDoc, out: Path) -> None:
    """Write ParsedDoc to a JSON file."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = dataclasses.asdict(doc)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _to_slug(name: str) -> str:
    """Convert a filename stem to a safe document ID."""
    slug = re.sub(r"[^\w\-]", "_", name)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "doc"
