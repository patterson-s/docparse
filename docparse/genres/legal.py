"""Legal-act genre (stub).

Registered so the genre system is already extensible to legislation, and so an
explicit `?genre=legal_act` request routes correctly today. The full layout
(parts -> articles -> schedules) is a future drop-in; for now it degrades to a
flat per-section vault using the academic-style writer.
"""

from __future__ import annotations

from pathlib import Path

from .base import GenreHandler, GenreConfig
from ..models import DocProfile, StructuredDoc
from ..vault_builder import make_slug, _write_main_md, _write_bibliographic_md


class LegalActGenre(GenreHandler):
    id = "legal_act"
    label = "Legal act"
    config = GenreConfig(
        chunk_window=250,
        chunk_overlap=40,
        extra_discard_patterns=[r"^Government Gazette.*$", r"^No\.\s*\d+ of \d{4}$"],
        plan_hint=(
            "This is a LEGAL ACT / statute. Identify parts, chapters, sections "
            "(numbered articles), schedules, and definitions. Keep the enacting "
            "formula and section numbering intact. Subsections are level 2."
        ),
    )

    def confidence(self, profile: DocProfile, sample: str) -> float:
        dt = (profile.doc_type or "").lower()
        if "legal" in dt or "act" in dt:
            return 0.9
        if "whereas" in sample[:3000].lower() or "enacted by" in sample[:3000].lower():
            return 0.6
        return 0.0

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

        title = meta.get("title") or Path(sdoc.filename).stem
        authors = meta.get("authors") or []
        year = meta.get("year")
        slug = make_slug(authors, year, title)

        _write_main_md(
            out_dir, slug, title, authors, year, meta.get("source"), meta.get("doi"),
            sdoc.filename, sdoc.raw_markdown, serper,
        )
        _write_bibliographic_md(
            out_dir, title, authors, year, meta.get("source") or "", meta.get("doi") or "",
            slug, sdoc.filename, serper,
        )

        # Per-section files (articles / parts) — fuller layout is a future task.
        for i, sec in enumerate(sdoc.sections, 1):
            safe = re.sub(r"[^\\w\\-]", "_", sec.section_id)[:60].strip("_") or f"sec_{i}"
            (out_dir / f"{i:02d}_{safe}.md").write_text(
                f"# {sec.label or sec.section_id}\n\n{sec.content}\n", encoding="utf-8"
            )
