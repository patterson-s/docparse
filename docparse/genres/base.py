"""Genre base classes + the per-genre processing config."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from ..models import DocProfile, StructuredDoc


@dataclass
class GenreConfig:
    """Knobs a genre applies to the shared pipeline.

    chunk_window / chunk_overlap - passage size used by chunker
    extra_discard_patterns       - regexes added to noise_filter (e.g. book
                                   publisher/copyright front+back matter)
    plan_hint                    - genre-specific instruction appended to the
                                   structurer's plan prompt (e.g. "group into
                                   chapters")
    """

    chunk_window: int = 300
    chunk_overlap: int = 50
    extra_discard_patterns: list[str] = None  # type: ignore[assignment]
    plan_hint: str = ""

    def __post_init__(self):
        if self.extra_discard_patterns is None:
            self.extra_discard_patterns = []


@dataclass
class VaultFile:
    """One file in a vault entry (relative path + content)."""

    rel_path: str
    content: str


class GenreHandler:
    """Base class for a document-genre plugin.

    A genre bundles: routing confidence, a processing config (chunk size,
    extra boilerplate to discard, structurer hint), and a vault layout.
    """

    id: str = "base"
    label: str = "Base"
    config: GenreConfig = None  # type: ignore[assignment]

    def __init__(self):
        # Clone the class-level config so instance mutations never leak.
        base = self.config or GenreConfig()
        self.config = GenreConfig(
            chunk_window=base.chunk_window,
            chunk_overlap=base.chunk_overlap,
            extra_discard_patterns=list(base.extra_discard_patterns or []),
            plan_hint=base.plan_hint,
        )

    def confidence(self, profile: DocProfile, sample: str) -> float:
        """0..1 — how sure this handler is the doc belongs to it.

        Used by GenreRouter only when no explicit genre override is given.
        """
        return 0.0

    def entry_name(self, sdoc: StructuredDoc, metadata: dict | None = None) -> str:
        """Slug for the vault folder. Default: filename stem."""
        from ..parser import _to_slug

        return _to_slug(Path(sdoc.filename).stem)

    def build_vault(
        self,
        sdoc: StructuredDoc,
        out_dir: Path,
        metadata: dict | None = None,
        serper: dict | None = None,
    ) -> None:
        """Write this genre's vault layout into out_dir (a fresh dir)."""
        raise NotImplementedError


def build_discard_regex(patterns: list[str]) -> list[re.Pattern[str]]:
    """Compile a genre's extra discard patterns once."""
    return [re.compile(p, re.IGNORECASE) for p in patterns]
