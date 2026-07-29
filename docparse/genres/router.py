"""Genre router: chooses a handler from a DocProfile, or honours an override."""

from __future__ import annotations

from .base import GenreHandler
from .academic import AcademicArticleGenre
from .book import BookGenre
from .legal import LegalActGenre
from ..models import DocProfile, StructuredDoc

# Order matters only as a tie-break; routing is by highest confidence.
_REGISTRY: dict[str, type] = {
    AcademicArticleGenre.id: AcademicArticleGenre,
    BookGenre.id: BookGenre,
    LegalActGenre.id: LegalActGenre,
}


def register_genre(handler_cls: type) -> None:
    """Register a new genre handler class (drop-in extensibility)."""
    _REGISTRY[handler_cls.id] = handler_cls


def available_genres() -> list[str]:
    return sorted(_REGISTRY)


def get_handler(genre_id: str | None) -> GenreHandler:
    """Return a handler by id, defaulting to academic_article if unknown/None."""
    if genre_id is None:
        return AcademicArticleGenre()
    cls = _REGISTRY.get(genre_id)
    if cls is None:
        return AcademicArticleGenre()
    return cls()


def route_genre(profile: DocProfile, sample: str, override: str | None = None) -> tuple[GenreHandler, str]:
    """Return (handler, resolved_genre_id).

    If `override` is given and known, it wins. Otherwise pick the highest
    confidence handler. Ties / low-confidence default to academic_article.
    """
    if override:
        return get_handler(override), override

    best: GenreHandler | None = None
    best_score = -1.0
    for cls in _REGISTRY.values():
        h = cls()
        score = h.confidence(profile, sample)
        if score > best_score:
            best_score = score
            best = h
    handler = best or AcademicArticleGenre()
    return handler, handler.id
