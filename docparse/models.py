"""Data models for parsed documents."""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Raw parse models ─────────────────────────────────────────────────────────

@dataclass
class Section:
    level: int              # 1=section, 2=subsection, 3=sub-subsection
    title: str
    content: str
    heading_path: list[str] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str           # "{doc_id}_{position:04d}"
    document_id: str
    chunk_type: str         # "full_doc" | "section" | "passage"
    heading_path: list[str]
    content: str
    token_count: int        # approximate (word_count * 1.3)
    position: int
    language: str = ""      # populated in structured mode


@dataclass
class ParsedDoc:
    document_id: str
    filename: str
    source_format: str      # "pdf" | "docx" | "md"
    metadata: dict
    sections: list[Section]
    chunks: list[Chunk]
    raw_markdown: str
    parsed_at: str


# ── Structured parse models ───────────────────────────────────────────────────

@dataclass
class DocProfile:
    doc_type: str               # "legal_act" | "contract" | "report" | "academic_paper" | etc.
    languages: list[str]        # ["English", "isiXhosa"]
    structure_pattern: str      # "bilingual_alternating" | "monolingual" | "bilingual_parallel" | etc.
    structure_notes: str        # 1-2 sentence description
    estimated_sections: list[dict] = field(default_factory=list)  # [{label, language}]


@dataclass
class StructuredSection:
    section_id: str     # slug, e.g. "chapter_1_en"
    label: str          # human label, e.g. "Chapter 1"
    language: str       # "en" | "xhosa" | "both" | "unknown"
    level: int          # 1=top, 2=sub
    content: str


@dataclass
class StructuredDoc:
    document_id: str
    filename: str
    profile: DocProfile
    sections: list[StructuredSection]
    chunks: list[Chunk]
    raw_markdown: str
    parsed_at: str
