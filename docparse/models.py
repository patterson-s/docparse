"""Data models for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Section:
    level: int              # 1=section, 2=subsection, 3=sub-subsection
    title: str
    content: str
    heading_path: list[str] = field(default_factory=list)  # breadcrumb from root


@dataclass
class Chunk:
    chunk_id: str           # "{doc_id}_{position:04d}"
    document_id: str
    chunk_type: str         # "full_doc" | "section" | "passage"
    heading_path: list[str]
    content: str
    token_count: int        # approximate (word_count * 1.3)
    position: int


@dataclass
class ParsedDoc:
    document_id: str        # slug derived from filename stem
    filename: str
    source_format: str      # "pdf" | "docx" | "md"
    metadata: dict
    sections: list[Section]
    chunks: list[Chunk]
    raw_markdown: str
    parsed_at: str          # ISO timestamp
