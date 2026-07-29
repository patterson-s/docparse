"""Academic article genre (the default).

Reuses the proven split_content + Obsidian layout from vault_builder, but
behind the GenreHandler interface so it slots into the router like any other
genre. Abstract / body / references / bibliographic.md per article.
"""

from __future__ import annotations

from pathlib import Path

from .base import GenreHandler, GenreConfig
from ..models import DocProfile, StructuredDoc
from ..vault_builder import split_content, make_slug, _write_main_md, _write_bibliographic_md


class AcademicArticleGenre(GenreHandler):
    id = "academic_article"
    label = "Academic article"
    config = GenreConfig(chunk_window=300, chunk_overlap=50)

    def confidence(self, profile: DocProfile, sample: str) -> float:
        dt = (profile.doc_type or "").lower()
        if any(k in dt for k in ("academic", "paper", "report", "article")):
            return 0.9
        # Default fallback — most ingested docs are articles.
        return 0.5

    def entry_name(self, sdoc: StructuredDoc, metadata: dict | None = None) -> str:
        meta = metadata or {}
        title = meta.get("title") or Path(sdoc.filename).stem
        authors = meta.get("authors") or []
        year = meta.get("year")
        return make_slug(authors, year, title)

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
        journal = meta.get("source")
        doi = meta.get("doi")
        slug = self.entry_name(sdoc, meta)
        full_md = sdoc.raw_markdown

        _write_main_md(
            out_dir, slug, title, authors, year, journal, doi,
            sdoc.filename, full_md, serper,
        )

        abstract, body, references = split_content(full_md)
        (out_dir / "abstract.md").write_text(
            abstract + "\n" if abstract else "", encoding="utf-8"
        )
        (out_dir / "body.md").write_text(
            body + "\n" if body else "", encoding="utf-8"
        )
        (out_dir / "references.md").write_text(
            references + "\n" if references else "", encoding="utf-8"
        )
        _write_bibliographic_md(
            out_dir, title, authors, year, journal or "", doi or "",
            slug, sdoc.filename, serper,
        )
