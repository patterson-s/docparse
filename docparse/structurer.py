"""Agentic structured post-processing pipeline.

Two LLM phases:
  1. Survey  — understand doc type, languages, structure pattern
  2. Plan    — map exact line ranges to labeled, language-tagged sections

Then deterministic execution: slice lines → StructuredDoc.

Output builders produce combined markdown, per-language markdown, and JSON.
"""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from .models import DocProfile, StructuredSection, StructuredDoc, Chunk
from .chunker import build_structured_chunks

_MAX_RETRIES = 3

# ── LLM call helper ───────────────────────────────────────────────────────────

def _call_json(client, model: str, system: str, user: str) -> dict:
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                raise
    return {}


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


# ── Phase 1: Survey ───────────────────────────────────────────────────────────

_SURVEY_SYSTEM = """Analyze the opening of this document and identify its key characteristics.

Return JSON with exactly these keys:
  doc_type           (string) — e.g. "legal_act", "contract", "report", "academic_paper", "policy_brief"
  languages          (list of strings) — language names present, e.g. ["English", "isiXhosa"]
  structure_pattern  (string) — one of: "monolingual" | "bilingual_alternating" | "bilingual_parallel" | "multilingual"
  structure_notes    (string) — 1-2 sentences describing how the document is organized
  estimated_sections (list) — array of {"label": str, "language": str} for the top-level sections you can infer

For structure_pattern:
  bilingual_alternating = full sections repeat back-to-back in each language (e.g. Chapter 1 EN then Chapter 1 XHO)
  bilingual_parallel    = two-column or interleaved sentence-by-sentence
  monolingual           = single language throughout
  multilingual          = 3+ languages"""


def _survey(client, model: str, lines: list[str]) -> DocProfile:
    sample = _numbered(lines[:250])
    data = _call_json(client, model, _SURVEY_SYSTEM, sample)
    return DocProfile(
        doc_type=data.get("doc_type", "unknown"),
        languages=data.get("languages", []),
        structure_pattern=data.get("structure_pattern", "unknown"),
        structure_notes=data.get("structure_notes", ""),
        estimated_sections=data.get("estimated_sections", []),
    )


# ── Phase 2: Structure Plan ───────────────────────────────────────────────────

def _build_plan_system(profile: DocProfile) -> str:
    lang_list = ", ".join(f'"{l}"' for l in profile.languages) or '"unknown"'
    return f"""You are structuring a {profile.doc_type} document.
Languages present: {", ".join(profile.languages)}.
Structure pattern: {profile.structure_pattern}.
Notes: {profile.structure_notes}

The document text has line numbers prefixed (e.g. "1: text").
Identify ALL top-level sections with their exact line ranges.

Rules:
- Every line must be in exactly one section — no gaps, no overlaps
- section_id: lowercase slug, e.g. "chapter_1_en", "preamble_xhosa", "toc"
- language: one of {lang_list} or "both" (for table of contents, cover page, etc.)
  Use short codes: "en", "xhosa", "fr", etc. — match what you see in the languages list but lowercase
- level: 1 for top-level sections, 2 for subsections
- start_line and end_line are 1-indexed and inclusive

Return JSON: {{"sections": [{{"label": str, "section_id": str, "language": str, "level": int, "start_line": int, "end_line": int}}, ...]}}"""


def _fill_gaps(sections: list[dict], total_lines: int) -> list[dict]:
    if not sections:
        return [{"label": "", "section_id": "body", "language": "unknown", "level": 1,
                 "start_line": 1, "end_line": total_lines}]

    sections = sorted(sections, key=lambda s: s["start_line"])
    filled: list[dict] = []
    cursor = 1

    for s in sections:
        if s["start_line"] > cursor:
            filled.append({"label": "", "section_id": f"unlabeled_{cursor}",
                           "language": "unknown", "level": 1,
                           "start_line": cursor, "end_line": s["start_line"] - 1})
        filled.append(s)
        cursor = max(cursor, s["end_line"] + 1)

    if cursor <= total_lines:
        filled.append({"label": "", "section_id": f"unlabeled_{cursor}",
                       "language": "unknown", "level": 1,
                       "start_line": cursor, "end_line": total_lines})
    return filled


def _plan(client, model: str, lines: list[str], profile: DocProfile) -> list[dict]:
    system = _build_plan_system(profile)
    data = _call_json(client, model, system, _numbered(lines))
    raw = data.get("sections", [])
    return _fill_gaps(raw, len(lines))


# ── Phase 3: Execute ──────────────────────────────────────────────────────────

def _slug(text: str) -> str:
    s = re.sub(r"[^\w\s-]", "", text.lower())
    s = re.sub(r"[\s-]+", "_", s).strip("_")
    return s or "section"


def _strip_bom(line: str) -> str:
    return line.lstrip("﻿")


def _execute(lines: list[str], plan: list[dict]) -> list[StructuredSection]:
    sections: list[StructuredSection] = []
    for s in plan:
        start = max(1, s["start_line"])
        end = min(len(lines), s["end_line"])
        if start > end:
            continue
        content_lines = [_strip_bom(l) for l in lines[start - 1 : end]]
        # Drop leading markdown heading line if it duplicates the label
        if content_lines and re.match(r"^#{1,6}\s+\S", content_lines[0].strip()):
            content_lines = content_lines[1:]
        content = "\n".join(content_lines).strip()
        if not content:
            continue

        section_id = s.get("section_id") or _slug(s.get("label", "section"))
        sections.append(StructuredSection(
            section_id=section_id,
            label=s.get("label", ""),
            language=s.get("language", "unknown"),
            level=max(1, min(2, s.get("level", 1))),
            content=content,
        ))
    return sections


# ── Orchestration ─────────────────────────────────────────────────────────────

def structure(
    raw_markdown: str,
    filename: str,
    document_id: str,
    model: str = "mistral-medium-latest",
    api_key: str = "",
) -> StructuredDoc:
    """Run survey → plan → execute on OCR markdown. Returns a StructuredDoc."""
    from mistralai import Mistral
    client = Mistral(api_key=api_key)

    lines = raw_markdown.splitlines()

    profile = _survey(client, model, lines)
    plan = _plan(client, model, lines, profile)
    sections = _execute(lines, plan)
    chunks = build_structured_chunks(document_id, sections, raw_markdown)

    return StructuredDoc(
        document_id=document_id,
        filename=filename,
        profile=profile,
        sections=sections,
        chunks=chunks,
        raw_markdown=raw_markdown,
        parsed_at=datetime.now(timezone.utc).isoformat(),
    )


# ── Output builders ───────────────────────────────────────────────────────────

def _lang_slug(lang: str) -> str:
    """Normalize language to a short lowercase slug for filenames."""
    mapping = {
        "english": "en", "isixhosa": "xhosa", "xhosa": "xhosa",
        "french": "fr", "afrikaans": "af", "zulu": "zu",
        "portuguese": "pt", "spanish": "es", "arabic": "ar",
    }
    return mapping.get(lang.lower(), lang.lower().replace(" ", "_"))


def _lang_display(lang: str) -> str:
    """Short uppercase display label for headings, e.g. 'EN', 'XHO'."""
    slug = _lang_slug(lang)
    short = {"en": "EN", "xhosa": "XHO", "fr": "FR", "af": "AF",
             "zu": "ZU", "pt": "PT", "es": "ES", "ar": "AR"}
    return short.get(slug, slug.upper()[:3])


def to_combined_markdown(doc: StructuredDoc) -> str:
    """All sections in one file with [LANG] suffix on headings."""
    lines: list[str] = []

    # YAML frontmatter
    langs = ", ".join(doc.profile.languages)
    lines += [
        "---",
        f"document_id: {doc.document_id}",
        f"doc_type: {doc.profile.doc_type}",
        f"languages: {langs}",
        f"structure_pattern: {doc.profile.structure_pattern}",
        f"parsed_at: {doc.parsed_at}",
        "---",
        "",
    ]

    for s in doc.sections:
        hashes = "#" * s.level
        lang_tag = f" [{_lang_display(s.language)}]" if s.language not in ("both", "unknown", "") else ""
        label = s.label or s.section_id
        lines.append(f"{hashes} {label}{lang_tag}")
        lines.append("")
        lines.append(s.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_language_markdown(doc: StructuredDoc, lang_slug: str) -> str:
    """Sections for a single language (plus 'both'/'unknown') — no [LANG] suffix."""
    # Normalise incoming slug
    norm = _lang_slug(lang_slug)

    included = [
        s for s in doc.sections
        if _lang_slug(s.language) == norm or s.language in ("both", "unknown")
    ]
    if not included:
        return ""

    lines: list[str] = []
    lines += [
        "---",
        f"document_id: {doc.document_id}",
        f"language: {lang_slug}",
        f"parsed_at: {doc.parsed_at}",
        "---",
        "",
    ]

    for s in included:
        hashes = "#" * s.level
        label = s.label or s.section_id
        lines.append(f"{hashes} {label}")
        lines.append("")
        lines.append(s.content)
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def to_structured_json(doc: StructuredDoc) -> str:
    data = dataclasses.asdict(doc)
    return json.dumps(data, indent=2, ensure_ascii=False)


def detected_language_slugs(doc: StructuredDoc) -> list[str]:
    """Return unique non-'both' language slugs present in the sections."""
    seen: set[str] = set()
    result: list[str] = []
    for s in doc.sections:
        if s.language not in ("both", "unknown", ""):
            slug = _lang_slug(s.language)
            if slug not in seen:
                seen.add(slug)
                result.append(slug)
    return result
