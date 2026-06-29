"""Chunking logic: full_doc + section + passage chunks.

Ported from LiteratureTool/ingestion/chunker.py (_strip_markdown, _sliding_window)
and adapted for the standalone ParsedDoc model.
"""

from __future__ import annotations

import re

from .models import Section, Chunk

PASSAGE_WORDS = 300
PASSAGE_OVERLAP_WORDS = 50


def build_chunks(doc_id: str, sections: list[Section], raw_markdown: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    pos = 0

    # 1. Full-doc chunk (stripped plain text)
    full_text = _strip_markdown(raw_markdown)
    chunks.append(Chunk(
        chunk_id=f"{doc_id}_{pos:04d}",
        document_id=doc_id,
        chunk_type="full_doc",
        heading_path=[],
        content=full_text,
        token_count=_approx_tokens(full_text),
        position=pos,
    ))
    pos += 1

    # 2. Section chunks (one per detected section)
    for section in sections:
        if not section.content.strip():
            continue
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{pos:04d}",
            document_id=doc_id,
            chunk_type="section",
            heading_path=section.heading_path,
            content=section.content,
            token_count=_approx_tokens(section.content),
            position=pos,
        ))
        pos += 1

    # 3. Passage chunks (sliding window over stripped full text)
    # Build a heading-path lookup by cumulative character offset across sections
    offset_map = _build_section_offset_map(sections)

    passages = _sliding_window(full_text, PASSAGE_WORDS, PASSAGE_OVERLAP_WORDS)
    for passage_text, char_offset in passages:
        heading_path = _heading_path_at_offset(offset_map, char_offset)
        chunks.append(Chunk(
            chunk_id=f"{doc_id}_{pos:04d}",
            document_id=doc_id,
            chunk_type="passage",
            heading_path=heading_path,
            content=passage_text,
            token_count=_approx_tokens(passage_text),
            position=pos,
        ))
        pos += 1

    return chunks


def _approx_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def _strip_markdown(markdown: str) -> str:
    text = markdown
    text = re.sub(r"```[\s\S]*?```", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"!\[.*?\]\(.*?\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\(.*?\)", r"\1", text)
    text = re.sub(r"[*_]{1,3}([^*_]+)[*_]{1,3}", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _sliding_window(
    text: str,
    window_words: int,
    overlap_words: int,
) -> list[tuple[str, int]]:
    words = text.split()
    if not words:
        return []

    step = window_words - overlap_words
    results: list[tuple[str, int]] = []
    i = 0

    while i < len(words):
        window = words[i : i + window_words]
        passage = " ".join(window)
        char_offset = len(" ".join(words[:i]))
        results.append((passage, char_offset))
        if i + window_words >= len(words):
            break
        i += step

    return results


def _build_section_offset_map(sections: list[Section]) -> list[tuple[int, list[str]]]:
    """Map cumulative stripped-text char offsets to heading_path for each section."""
    result: list[tuple[int, list[str]]] = []
    offset = 0
    for s in sections:
        stripped = _strip_markdown(s.content)
        result.append((offset, s.heading_path))
        offset += len(stripped) + 1
    return result


def _heading_path_at_offset(
    offset_map: list[tuple[int, list[str]]], char_offset: int
) -> list[str]:
    path: list[str] = []
    for pos, heading_path in offset_map:
        if pos <= char_offset:
            path = heading_path
        else:
            break
    return path
