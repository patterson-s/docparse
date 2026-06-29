"""LLM-based heading detection using Mistral.

Adapted from LiteratureTool/latex_structuring/structurer.py _detect_sections()
and _fill_gaps(), but using Mistral chat instead of Cohere, and extended to
handle both academic papers and consulting/IDRC-style reports.
"""

from __future__ import annotations

import json
import re

from mistralai import Mistral

from .models import Section

_SYSTEM = """You are analyzing a document (OCR'd or converted to markdown) with line numbers prefixed.
Detect the heading hierarchy and return a complete section tree.

Recognize headings from:
- Markdown markers (# H1, ## H2, ### H3)
- Numbered headings (1 Introduction, 2.1 Methods, 3.2.1 Sub-section)
- IDRC/consulting report sections (Project Highlights, Research Outputs, Project Outcomes,
  Objectives & Ratings, Financial Performance, Director Comments, Executive Summary,
  Key Findings, Recommendations, Background, Methodology, Conclusion)
- Academic paper sections (Abstract, Introduction, Literature Review, Methods, Results,
  Discussion, Conclusion, Acknowledgments, References)
- Any line that is short, isolated, and clearly titles content below it

Rules:
- Assign level 1 to top-level sections, 2 to subsections, 3 to sub-subsections
- Return sections covering the ENTIRE document, non-overlapping, with no gaps
- Every line must belong to exactly one section
- If content appears before the first real heading, give it level 1 with an empty title
- start_line and end_line are 1-indexed and inclusive

Return JSON: {"sections": [{"level": 1, "title": "...", "start_line": N, "end_line": M}, ...]}"""

_MAX_RETRIES = 3


def _numbered(lines: list[str]) -> str:
    return "\n".join(f"{i + 1}: {line}" for i, line in enumerate(lines))


def _call_json(client: Mistral, model: str, lines: list[str]) -> dict:
    user = _numbered(lines)
    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
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


def _fill_gaps(sections: list[dict], total_lines: int) -> list[dict]:
    """Ensure complete coverage — no lines silently dropped."""
    if not sections:
        return [{"level": 1, "title": "", "start_line": 1, "end_line": total_lines}]

    sections = sorted(sections, key=lambda s: s["start_line"])
    filled: list[dict] = []
    cursor = 1

    for s in sections:
        if s["start_line"] > cursor:
            filled.append({"level": 1, "title": "", "start_line": cursor, "end_line": s["start_line"] - 1})
        filled.append(s)
        cursor = max(cursor, s["end_line"] + 1)

    if cursor <= total_lines:
        filled.append({"level": 1, "title": "", "start_line": cursor, "end_line": total_lines})

    return filled


def _build_heading_paths(raw: list[dict]) -> list[dict]:
    """Add heading_path breadcrumbs based on level hierarchy."""
    current: dict[int, str | None] = {1: None, 2: None, 3: None}
    result = []

    for s in raw:
        level = max(1, min(3, s.get("level", 1)))
        title = s.get("title", "").strip()
        current[level] = title
        for l in range(level + 1, 4):
            current[l] = None
        path = [current[l] for l in range(1, level + 1) if current[l]]
        result.append({**s, "level": level, "title": title, "heading_path": path})

    return result


def detect(lines: list[str], model: str = "mistral-medium-latest", api_key: str = "") -> list[Section]:
    """Detect sections in a list of document lines. Returns Section objects with heading_path."""
    if not lines:
        return []

    client = Mistral(api_key=api_key)
    data = _call_json(client, model, lines)
    raw_sections = _fill_gaps(data.get("sections", []), len(lines))
    raw_sections = _build_heading_paths(raw_sections)

    sections: list[Section] = []
    for s in raw_sections:
        start = max(1, s["start_line"])
        end = min(len(lines), s["end_line"])
        if start > end:
            continue

        content_lines = lines[start - 1 : end]
        # Drop the heading line itself if it's a markdown heading (already stored as title)
        if content_lines and re.match(r"^#{1,6}\s+\S", content_lines[0].strip()):
            content_lines = content_lines[1:]

        content = "\n".join(content_lines).strip()
        if not content:
            continue

        sections.append(Section(
            level=s["level"],
            title=s["title"],
            content=content,
            heading_path=s["heading_path"],
        ))

    return sections
