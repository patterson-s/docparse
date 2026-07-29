"""Top-level processing pipeline used by the API / messaging layers.

This is the single place that wires genres and providers together. Everything
upstream (FastAPI, Telegram, CLI) just calls `run_pipeline(...)` with an
already-resolved DocumentSource, genre override, and provider ids.

Stages:
  1. OCR (OcrProvider)            -> raw_markdown
  2. Survey (ChatProvider)        -> DocProfile
  3. Genre routing                -> handler + config (window/overlap/plan_hint)
  4. Structure plan (ChatProvider)-> sections
  5. Chunk                        -> chunks (genre window/overlap)
  6. Metadata (ChatProvider)      -> front matter
  7. Vault layout (handler)       -> files written to out_dir (or returned)
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from . import providers
from .providers import DocumentSource
from . import structurer
from . import metadata_extractor
from . import genres
from .models import StructuredDoc


def run_pipeline(
    source: DocumentSource,
    *,
    filename: Optional[str] = None,
    model: str = "mistral-medium-latest",
    api_key: str = "",
    chat_provider=None,
    ocr_provider=None,
    genre_override: Optional[str] = None,
    return_vault: bool = False,
    vault_dir: Optional[Path] = None,
    serper_key: str = "",
) -> dict:
    """Run the full structured pipeline and return a result dict.

    Returns:
      {
        "document_id", "filename", "genre",
        "profile": {...}, "sections": [...], "chunks": [...],
        "metadata": {...}, "raw_markdown": str,
        "vault_files": [{"rel_path", "content"}, ...]  # if return_vault
      }
    """
    filename = filename or source.filename

    from .parser import _to_slug
    from .chunker import build_structured_chunks

    # Resolve providers (Mistral default).
    if chat_provider is None:
        chat_provider = providers.get_chat_provider(api_key=api_key)
    elif isinstance(chat_provider, str):
        chat_provider = providers.get_chat_provider(chat_provider, api_key=api_key)
    if ocr_provider is None:
        ocr_provider = providers.get_ocr_provider(api_key=api_key)
    elif isinstance(ocr_provider, str):
        ocr_provider = providers.get_ocr_provider(ocr_provider, api_key=api_key)

    # 1. OCR
    raw_markdown = ocr_provider.extract(source)
    lines = raw_markdown.splitlines()

    # 2. Survey
    profile = structurer._survey(chat_provider, model, lines)

    # 3. Genre routing (use survey profile, fall back to override)
    handler, genre_id = genres.route_genre(profile, raw_markdown, override=genre_override)
    cfg = handler.config

    # 4. Structure plan (+ genre hint). Books are split deterministically (the
    #    generic LLM plan is unstable on long documents), so branch there.
    if genre_id == "book":
        from .genres import book as book_genre
        vault_files = book_genre.process_book(
            raw_markdown, filename, chat_provider=chat_provider, model=model,
            substructure=False,
        )
        doc_id = _to_slug(Path(filename).stem)
        # Synthesize a minimal StructuredDoc so downstream (chunks/result) works.
        sdoc = StructuredDoc(
            document_id=doc_id,
            filename=filename,
            profile=profile,
            sections=[],  # book uses per-chapter files instead
            chunks=[],
            raw_markdown=raw_markdown,
            parsed_at=datetime.now(timezone.utc).isoformat(),
        )
        sections = []
        chunks = []
    else:
        plan = structurer._plan(chat_provider, model, lines, profile, plan_hint=cfg.plan_hint)
        sections = structurer._execute(lines, plan)

        # 5. Chunk (genre window/overlap)
        doc_id = _to_slug(Path(filename).stem)
        chunks = build_structured_chunks(
            doc_id, sections, raw_markdown, window=cfg.chunk_window, overlap=cfg.chunk_overlap
        )
        sdoc = StructuredDoc(
            document_id=doc_id,
            filename=filename,
            profile=profile,
            sections=sections,
            chunks=chunks,
            raw_markdown=raw_markdown,
            parsed_at=datetime.now(timezone.utc).isoformat(),
        )

    # 6. Metadata
    metadata = metadata_extractor.extract(raw_markdown, model=model, chat_provider=chat_provider)

    result: dict = {
        "document_id": doc_id,
        "filename": filename,
        "genre": genre_id,
        "profile": dataclasses.asdict(profile),
        "sections": dataclasses.asdict(sdoc)["sections"],
        "chunks": dataclasses.asdict(sdoc)["chunks"],
        "metadata": metadata,
        "raw_markdown": raw_markdown,
    }

    # 7. Vault layout (genre-specific)
    if return_vault:
        out = vault_dir or Path.cwd() / "vault_out" / doc_id
        out.mkdir(parents=True, exist_ok=True)
        serper = {}
        if serper_key:
            from .vault_builder import enrich_via_serper
            serper = enrich_via_serper(
                metadata.get("doi"), metadata.get("title"), metadata.get("authors") or [], serper_key
            )
        if genre_id == "book":
            # Books already produced per-chapter vault files in stage 4.
            for f in vault_files:
                (out / f["rel_path"]).write_text(f["content"], encoding="utf-8")
        else:
            handler.build_vault(sdoc, out, metadata=metadata, serper=serper)
        result["vault_dir"] = str(out)

    return result
