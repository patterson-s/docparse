"""Tests for the book genre handler (no network — deterministic split).

Proves the whole-book -> per-chapter split works on the real Sending book OCR
markdown (front/back matter separated, PARTs and CHAPTERs detected) and on a
synthetic book.
"""

from __future__ import annotations

from pathlib import Path

from docparse.genres.book import split_book, process_book, BookGenre
from docparse.models import DocProfile, StructuredDoc


SENDING = Path(__file__).parent / "Sending_2015_PoliticsExpertiseCompeting" / \
    "Sending_2015_PoliticsExpertiseCompeting.md"


def _profile(doc_type: str = "book") -> DocProfile:
    return DocProfile(doc_type=doc_type, languages=["English"],
                      structure_pattern="monolingual", structure_notes="",
                      estimated_sections=[])


def test_split_sending_book_detects_parts_and_chapters():
    raw = SENDING.read_text(encoding="utf-8")
    bs = split_book(raw)
    # 3 parts; the book has Introduction + 5 chapters + Conclusion = 7 sections.
    assert bs.parts == ["PART I", "PART II", "PART III"], bs.parts
    assert len(bs.chapters) == 7, len(bs.chapters)
    titles = [c.title for c in bs.chapters]
    assert "Introduction" in titles
    assert "Competing for Authority" in titles
    assert "Conclusion" in titles
    assert "Diplomats, Lawyers, and the Emergence of International Rule" in titles
    # Front/back matter separated (NOT inside chapters).
    joined = "\n".join(c.text for c in bs.chapters)
    assert "Acknowledgments" not in joined
    assert "Bibliography" not in joined and "Index" not in joined


def test_split_sending_excludes_toc():
    raw = SENDING.read_text(encoding="utf-8")
    bs = split_book(raw)
    # No chapter whose title looks like a TOC entry ("Chapter N. Long Title 123").
    for ch in bs.chapters:
        assert "chapter" not in ch.title.lower()[:4], ch.title


def test_process_book_writes_vault_files():
    raw = SENDING.read_text(encoding="utf-8")
    files = process_book(raw, "Sending_2015_PoliticsExpertiseCompeting.md")
    rels = {f["rel_path"] for f in files}
    assert "contents.md" in rels
    assert "bibliographic.md" in rels
    assert "00_front_matter.md" in rels
    assert "zz_back_matter.md" in rels
    # 7 per-chapter files (NN_<slug>.md), excluding 00_front_matter / zz_back_matter.
    import re
    chap_files = [f for f in files
                  if re.match(r"^\d{2}_.*\.md$", f["rel_path"])
                  and not f["rel_path"].startswith("00_")
                  and not f["rel_path"].startswith("zz_")]
    assert len(chap_files) == 7, [f["rel_path"] for f in chap_files]


def test_book_genre_confidence_on_book_profile():
    h = BookGenre()
    assert h.confidence(_profile("book"), "") >= 0.9
    # An article-like profile scores low.
    assert h.confidence(_profile("academic_paper"), "introduction\nmethods\nconclusion") < 0.5


def test_synthetic_book_split():
    raw = (
        "---\ntitle: T\n---\n"
        "# Foreword\nThanks.\n\n# Acknowledgments\nTo mom.\n\n"
        "# Contents\nChapter 1. One 3\nChapter 2. Two 9\n\n"
        "# PART I\n\n# Chapter One\nBody of one.\n\n# Chapter Two\nBody of two.\n\n"
        "# References\nSmith, 2020.\n\n# Index\nA, 1.\n"
    )
    bs = split_book(raw)
    assert "Foreword" in bs.front_matter and "Acknowledgments" in bs.front_matter
    assert [c.title for c in bs.chapters] == ["Chapter One", "Chapter Two"]
    assert "Smith, 2020" in bs.back_matter
