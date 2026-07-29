"""Tests for the Signal adapter (no signald / no network).

Covers the pure command-parsing helper and the shared vault-write path
(process_document) that the Signal handler delegates to.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from docparse.providers import (
    ChatProvider, OcrProvider, DocumentSource,
    register_chat_provider, register_ocr_provider,
)
from docparse.signal_bot import parse_control_message
from docparse.telegram_bot import VaultTarget, process_document


class _FakeOcr(OcrProvider):
    name = "fake-ocr"

    def extract(self, source: DocumentSource) -> str:
        return "# Abstract\nA.\n\n# Intro\nB.\n\n# References\n[1] X.\n"


class _FakeChat(ChatProvider):
    name = "fake-chat"

    def complete(self, messages, *, response_format=None, model=None, temperature=0.0):
        system = messages[0]["content"]
        if "Analyze the opening" in system:
            return {"doc_type": "academic_paper", "languages": ["English"],
                    "structure_pattern": "monolingual", "structure_notes": "",
                    "estimated_sections": []}
        if "Extract document metadata" in system:
            return {"title": "T", "authors": [], "year": None,
                    "abstract": None, "source": None, "doi": None}
        return {}


def _setup():
    register_ocr_provider("fake-ocr", _FakeOcr)
    register_chat_provider("fake-chat", _FakeChat)


def test_parse_control_message_commands():
    assert parse_control_message("/genre book") == ("genre", "book")
    assert parse_control_message("/vault research") == ("vault", "research")
    assert parse_control_message("/target /abs/path") == ("vault", "/abs/path")
    assert parse_control_message("/help") == ("help", None)
    assert parse_control_message("https://example.com/doc.pdf") == ("url", "https://example.com/doc.pdf")
    # unknown command -> ignore
    assert parse_control_message("/bogus") == ("ignore", None)
    # plain text -> doc (no special handling expected)
    assert parse_control_message("hello") == ("doc", None)


def test_signal_vault_routing_through_process_document():
    _setup()
    d = Path(tempfile.mkdtemp())
    md = d / "Article.md"
    md.write_text("# Abstract\nA.\n\n# Intro\nB.\n\n# References\n[1] X.\n")

    vault_root = Path(tempfile.mkdtemp())
    # Simulate /vault research then process an attachment at the resolved path.
    summary = process_document(
        md, api_base=None,
        vault_target=VaultTarget(root=vault_root, override="research"),
        chat_provider="fake-chat", ocr_provider="fake-ocr", genre="academic_article",
    )
    assert summary["vault_dir"] == str(vault_root / "research" / "Article")
    assert Path(summary["vault_dir"]).exists()


def test_signal_absolute_vault_override():
    _setup()
    d = Path(tempfile.mkdtemp())
    md = d / "Note.md"
    md.write_text("# A\nx\n\n# B\ny\n")
    vault_root = Path(tempfile.mkdtemp())
    target_dir = vault_root / "myvault"
    summary = process_document(
        md, api_base=None,
        vault_target=VaultTarget(root=vault_root, override=str(target_dir)),
        chat_provider="fake-chat", ocr_provider="fake-ocr",
    )
    assert summary["vault_dir"] == str(target_dir / "Note")
