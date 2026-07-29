"""Genre plugin system.

A genre is a bundle of (a) routing confidence, (b) processing config (chunk
size, extra boilerplate to discard, sectioning hint), and (c) a vault layout.

Adding a new genre = drop a module + register it. The pipeline asks the
GenreRouter for the best handler (or uses an explicit override) and then only
ever talks to the `GenreHandler` interface. Nothing else in docparse knows the
difference between an article, a book, and a legal act.
"""

from __future__ import annotations

from .base import GenreConfig, GenreHandler, VaultFile
from .academic import AcademicArticleGenre
from .book import BookGenre
from .legal import LegalActGenre
from .router import route_genre, register_genre, available_genres, get_handler

__all__ = [
    "GenreConfig",
    "GenreHandler",
    "VaultFile",
    "AcademicArticleGenre",
    "BookGenre",
    "LegalActGenre",
    "route_genre",
    "register_genre",
    "available_genres",
    "get_handler",
]
