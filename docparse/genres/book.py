"""Book genre — whole-book -> per-chapter vault.

Books are structured very differently from articles, and asking an LLM to plan
a 2,000+ line book in one shot produces unstable, over-segmented results (we saw
this on the article matrix: "stricter" prompts hallucinated 40+ sections). So
the book handler does the PART/CHAPTER split DETERMINISTICALLY from the OCR
markdown — regex + light heuristics — which is cheap, reproducible, and far more
reliable for long documents. Each chapter is then optionally re-structured
(its own subsections) by the chat provider if one is supplied.

Outputs (Obsidian vault layout for a book):
  bibliographic.md      - front matter (title/authors/year/source/doi)
  contents.md           - part + chapter index (the reading order)
  00_front_matter.md    - foreword / acknowledgments / preface (curated, kept)
  NN_<chapter-slug>.md  - one file per chapter (title + body; subsections if
                          substructuring is on)
  zz_back_matter.md     - references / notes / index (curated, kept)
  raw.md                - full OCR text, for reference
  metadata.json         - machine-readable

Front/back matter is *detected* (not silently dropped): we keep a curated
front_matter / back_matter file rather than discarding publisher boilerplate
entirely, but we never let it pollute the chapter sequence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .base import GenreHandler, GenreConfig
from ..models import DocProfile, StructuredDoc
from ..vault_builder import make_slug, _write_bibliographic_md


# ── Detection patterns ──────────────────────────────────────────────────────────

_PART_RE = re.compile(r"^\s*#?\s*(part\s+([IVXLCivxlc\d]+(?:\s*[.:-]\s*\w+)?))\s*$", re.IGNORECASE)
_CHAPTER_RE = re.compile(
    r"^\s*#?\s*(chapter\s+([IVXLCivxlc\d]+(?:\s*[.:-]\s*\w+)?))\b", re.IGNORECASE
)

# Front/back matter boundary headings (case-insensitive, allow leading #).
_FRONT_MARKERS = [
    r"foreword", r"preface", r"acknowledgements?", r"contents?",
    r"table of contents", r"list of (figures|tables|abbreviations|contributors)",
    r"abbreviations", r"glossary", r"series (foreword|editor)", r"general editor",
    r"dedication", r"about the (author|editor|book)", r"notes on (the|this) (text|book)",
]
_BACK_MARKERS = [
    r"references", r"bibliography", r"notes", r"index", r"appendix",
    r"author (index|bio)", r"about the (author|editor)", r"colophon",
]
_FRONT_RE = re.compile(r"^\s*#?\s*(" + "|".join(_FRONT_MARKERS) + r")\s*$", re.IGNORECASE)
_BACK_RE = re.compile(r"^\s*#?\s*(" + "|".join(_BACK_MARKERS) + r")\s*$", re.IGNORECASE)

_TITLE_HEADING_RE = re.compile(r"^\s*#{1,2}\s+(.+?)\s*$")


@dataclass
class ChapterBlock:
    part: str = ""               # e.g. "PART I"
    number: str = ""             # e.g. "I", "II", or "" for unnumbered
    title: str = ""              # chapter title (cleaned)
    text: str = ""               # body markdown (boilerplate stripped)


@dataclass
class BookStructure:
    front_matter: str = ""
    chapters: list = field(default_factory=list)   # list[ChapterBlock]
    back_matter: str = ""
    parts: list = field(default_factory=list)      # ordered part labels


def _clean_heading(line: str) -> str:
    m = _TITLE_HEADING_RE.match(line)
    txt = m.group(1) if m else line.strip("#").strip()
    return txt.replace("*", "").replace("_", "").strip()


def _is_italic_title(line: str) -> str | None:
    m = re.match(r"^\s*#{1,3}\s+\*(.+?)\*\s*$", line)
    return m.group(1).strip() if m else None


def _is_toc_line(line: str) -> bool:
    """A table-of-contents artifact: 'Chapter/Part N. Long Title 123'."""
    m = re.match(r"^\s*#?\s*(chapter|part)\s+[IVXLCivxlc\d]+\.\s+\S.*\d\s*$", line, re.IGNORECASE)
    if m:
        return True
    m2 = re.match(r"^\s*#?\s*(chapter|part)\s+[IVXLCivxlc\d]+\.\s+\S", line, re.IGNORECASE)
    if m2 and len(line.split()) > 6:
        return True
    return False


def _chapter_number(cm: re.Match) -> str:
    return _clean_heading(cm.group(2)) if cm.lastindex and cm.lastindex >= 2 else ""


def _strip_yaml(text: str) -> str:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            return text[end + 4:].strip()
    return text


def _strip_boilerplate(text: str) -> str:
    return _BOILERPLATE_RE.sub(r"\1", text).strip()


_BOILERPLATE_RE = re.compile(
    r"(^|\n)\s*(ISBN[-: ]?\S+|"
    r"All rights reserved|"
    r"Library of Congress Catalog|"
    r"Printed in |"
    r"Published by |"
    r"First (published|edition)|"
    r"This book is printed on |"
    r"©\s*\d{4}|"
    r"Copyright\s+©?\s*\d{4}|"
    r"\d{13}|\d{10})",
    re.IGNORECASE,
)


def split_book(raw_markdown: str) -> BookStructure:
    """Deterministically split a book's OCR markdown into front / chapters / back.

    Pure function (no network). Reproducible across runs and providers.
    """
    lines = raw_markdown.splitlines()
    struct = BookStructure()

    # Strip a leading YAML front-matter block (--- ... ---) if present.
    start = 0
    if lines and lines[0].strip() == "---":
        for i in range(1, len(lines)):
            if lines[i].strip() == "---":
                start = i + 1
                break
    lines = lines[start:]

    # Locate the Contents / Table of Contents block so we can skip the TOC when
    # searching for the real body start.
    contents_idx = None
    for i, ln in enumerate(lines):
        if re.match(r"^\s*#?\s*(table of )?contents\s*$", ln, re.IGNORECASE):
            contents_idx = i
            break

    # Body starts at the first REAL PART/CHAPTER heading AFTER the contents
    # block. Real headings carry a leading '#'. We deliberately do NOT start at
    # front-matter headings (Foreword/Acknowledgments) — those belong in
    # front_matter, which is everything before the first PART/CHAPTER.
    search_from = (contents_idx + 1) if contents_idx is not None else 0
    _REAL_HEAD_RE = re.compile(r"^\s*#\s+(part|chapter)\s", re.IGNORECASE)
    body_start = len(lines)
    for i in range(search_from, len(lines)):
        ln = lines[i]
        if _REAL_HEAD_RE.match(ln):
            body_start = i
            break

    struct.front_matter = _strip_yaml("\n".join(lines[:body_start]).strip())
    rest = lines[body_start:]

    # Back matter starts at the FIRST back marker in `rest` (so References /
    # Bibliography / Index close the chapter sequence rather than becoming a
    # spurious chapter).
    back_idx = len(rest)
    for i, ln in enumerate(rest):
        if _BACK_RE.match(ln):
            back_idx = i
            break
    body = rest[:back_idx]
    struct.back_matter = "\n".join(rest[back_idx:]).strip()

    # Walk the body, segmenting into chapters.
    cur_part = ""
    cur_ch: ChapterBlock | None = None
    buf: list[str] = []
    just_marker = False  # a chapter was just opened by a PART/CHAPTER marker

    def flush():
        nonlocal cur_ch, buf, just_marker
        if cur_ch is not None:
            cur_ch.text = _strip_boilerplate("\n".join(buf).strip())
            struct.chapters.append(cur_ch)
        cur_ch = None
        buf = []
        just_marker = False

    for ln in body:
        if _is_toc_line(ln):
            continue  # skip table-of-contents artifacts
        pm = _PART_RE.match(ln)
        cm = _CHAPTER_RE.match(ln)
        if pm:
            flush()
            cur_part = _clean_heading(ln).upper()
            if cur_part and cur_part not in struct.parts:
                struct.parts.append(cur_part)
            continue
        if cm:
            flush()
            cur_ch = ChapterBlock(part=cur_part, number=_chapter_number(cm))
            just_marker = True
            continue
        # Headings. A level-1 (#) title that immediately follows a PART/CHAPTER
        # marker is that chapter's own title (don't start a new chapter). A
        # standalone level-1 title starts a new chapter. Level-2 (##) headings
        # are subsections / chapter titles and never break the chapter.
        hm = _TITLE_HEADING_RE.match(ln)
        is_l1 = bool(re.match(r"^#\s", ln)) and not bool(re.match(r"^##\s", ln))
        if hm and is_l1 and not _FRONT_RE.match(ln) and not _is_toc_line(ln):
            if just_marker:
                just_marker = False  # consume: this # heading is the open chapter's title
            else:
                flush()
                cur_ch = ChapterBlock(part=cur_part)
        if cur_ch is not None:
            # Capture chapter title (any heading level); prefer an italic
            # subtitle (## *Title*).
            if hm and not cur_ch.title and not _is_toc_line(ln) and not _FRONT_RE.match(ln):
                it = _is_italic_title(ln)
                cur_ch.title = it if it else _clean_heading(ln)
            buf.append(ln)
        elif _FRONT_RE.match(ln):
            continue  # front-matter heading inside body (rare) — skip

    flush()

    # De-duplicate parts case-insensitively (TOC + real headings may both list them).
    seen: set[str] = set()
    deduped: list[str] = []
    for p in struct.parts:
        key = p.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    struct.parts = deduped

    return struct


def _slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", s.lower()).strip("_")
    return s[:60] or "chapter"


def process_book(
    raw_markdown: str,
    filename: str,
    *,
    chat_provider=None,
    model: str = "mistral-medium-latest",
    substructure: bool = False,
) -> list[dict]:
    """Split a book and return vault file dicts.

    If `substructure` and a `chat_provider` are given, each chapter is run
    through structurer.structure to produce its own subsections; otherwise the
    raw chapter text is written as-is.

    Returns a list of {"rel_path", "content"} (VaultFile-like).
    """
    struct = split_book(raw_markdown)
    title = Path(filename).stem
    files: list[dict] = []

    files.append({
        "rel_path": "bibliographic.md",
        "content": f"---\ntitle: {title}\nsource_file: {filename}\n---\n# {title}\n",
    })

    if struct.front_matter:
        files.append({"rel_path": "00_front_matter.md", "content": struct.front_matter})

    for i, ch in enumerate(struct.chapters, 1):
        label = ch.title or ch.number or f"chapter_{i}"
        head = f"# {label}"
        if ch.part:
            head += f"\n\n> {ch.part}"
        if substructure and chat_provider is not None:
            from .. import structurer
            try:
                sub = structurer.structure(
                    ch.text, filename, f"{Path(filename).stem}_{i}",
                    model=model, chat_provider=chat_provider,
                )
                body = structurer.to_combined_markdown(sub)
            except Exception:
                body = ch.text
        else:
            body = ch.text
        files.append({"rel_path": f"{i:02d}_{_slug(label)}.md", "content": f"{head}\n\n{body}\n"})

    # Contents index (parts + chapters, in reading order).
    idx = ["# Contents", ""]
    part_cursor = None
    n = 0
    for ch in struct.chapters:
        n += 1
        if ch.part and ch.part != part_cursor:
            part_cursor = ch.part
            idx.append(f"\n## {part_cursor}")
        idx.append(f"{n}. {ch.title or ch.number or f'chapter_{n}'}")
    files.append({"rel_path": "contents.md", "content": "\n".join(idx) + "\n"})

    if struct.back_matter:
        files.append({"rel_path": "zz_back_matter.md", "content": struct.back_matter})

    files.append({"rel_path": "raw.md", "content": raw_markdown})
    return files


class BookGenre(GenreHandler):
    id = "book"
    label = "Book"
    config = GenreConfig(
        chunk_window=150,
        chunk_overlap=30,
        extra_discard_patterns=[],
        plan_hint=(
            "This is a BOOK. The pipeline will split it into PARTS and CHAPTERS "
            "deterministically; do not attempt to plan the whole book as one "
            "document."
        ),
    )

    def confidence(self, profile: DocProfile, sample: str) -> float:
        dt = (profile.doc_type or "").lower()
        if "book" in dt:
            return 0.95
        score = 0.0
        lowered = sample.lower()
        if re.search(r"\bchapter\s+[ivxlc\d]", lowered):
            score += 0.5
        if "table of contents" in lowered[:4000] or "contents" in lowered[:2000]:
            score += 0.2
        if "isbn" in lowered[:2000]:
            score += 0.2
        if re.search(r"\bpart\s+[ivxlc\d]", lowered):
            score += 0.1
        return min(score, 0.95)

    def build_vault(
        self,
        sdoc: StructuredDoc,
        out_dir: Path,
        metadata: dict | None = None,
        serper: dict | None = None,
    ) -> None:
        meta = metadata or {}
        serper = serper or {}
        out_dir.mkdir(parents=True, exist_ok=True)

        files = process_book(sdoc.raw_markdown, sdoc.filename, substructure=False)
        for f in files:
            (out_dir / f["rel_path"]).write_text(f["content"], encoding="utf-8")

        authors = meta.get("authors") or []
        year = meta.get("year")
        slug = make_slug(authors, year, meta.get("title") or Path(sdoc.filename).stem)
        _write_bibliographic_md(
            out_dir, meta.get("title") or Path(sdoc.filename).stem, authors, year,
            meta.get("source") or "", meta.get("doi") or "", slug, sdoc.filename, serper,
        )
