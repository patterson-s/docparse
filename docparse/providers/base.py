"""Provider base classes and the DocumentSource transport type.

A provider is the *only* place that talks to a specific vendor SDK. Everything
else in docparse depends on the `ChatProvider` / `OcrProvider` interfaces, never
on Mistral directly. This is what makes "experiment with different APIs" a config
change rather than a fork.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class ProviderError(RuntimeError):
    """Raised when a provider call fails (after internal retries)."""


@dataclass
class DocumentSource:
    """A document to OCR, independent of where the bytes live.

    `url` is preferred (no upload round-trip for cloud OCR). `content` is used
    for uploaded bytes. Exactly one of the two should be set.
    """

    filename: str
    url: Optional[str] = None
    content: Optional[bytes] = None

    @staticmethod
    def from_path(path: str | Path) -> "DocumentSource":
        p = Path(path)
        return DocumentSource(filename=p.name, content=p.read_bytes())

    @staticmethod
    def from_url(url: str, filename: str = "") -> "DocumentSource":
        name = filename or Path(url.split("?")[0]).name or "document"
        return DocumentSource(filename=name, url=url)

    @staticmethod
    def from_bytes(content: bytes, filename: str) -> "DocumentSource":
        return DocumentSource(filename=filename, content=content)


class ChatProvider:
    """Structured-JSON chat completion. Implementations parse the model's
    content into a dict and retry transparently."""

    name: str = "base"

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict:
        raise NotImplementedError

    # Convenience used by the survey/plan prompts: a single system+user turn.
    def complete_json(
        self,
        system: str,
        user: str,
        *,
        response_format: Optional[dict] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict:
        return self.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format=response_format,
            model=model,
            temperature=temperature,
        )


class OcrProvider:
    """Document -> markdown. Implementations decide upload vs URL fetch."""

    name: str = "base"

    def extract(self, source: DocumentSource) -> str:
        raise NotImplementedError
