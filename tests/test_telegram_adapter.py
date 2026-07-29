"""Tests for the Telegram adapter (no Telegram client / no network).

Covers VaultTarget resolution and the core process_document write path using
the in-process pipeline (api_base=None) with fake providers.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from docparse.providers import (
    ChatProvider, OcrProvider, DocumentSource,
    register_chat_provider, register_ocr_provider,
)
from docparse.telegram_bot import VaultTarget, process_document


class _FakeOcr(OcrProvider):
    name = "fake-ocr"

    def extract(self, source: DocumentSource) -> str:
        return (
            "---\ntitle: T\n---\n"
            "# PART I\n\n# Introduction\nIntro body.\n\n"
            "## *Subtitle*\nMore.\n\n# PART II\n\n# Chapter One\nBody one.\n\n"
            "# References\nSmith, 2020.\n"
        )


class _FakeChat(ChatProvider):
    name = "fake-chat"

    def complete(self, messages, *, response_format=None, model=None, temperature=0.0):
        system = messages[0]["content"]
        if "Analyze the opening" in system:
            return {"doc_type": "book", "languages": ["English"],
                    "structure_pattern": "monolingual", "structure_notes": "",
                    "estimated_sections": []}
        if "Extract document metadata" in system:
            return {"title": "T", "authors": [], "year": None,
                    "abstract": None, "source": None, "doi": None}
        return {}


def _setup():
    register_ocr_provider("fake-ocr", _FakeOcr)
    register_chat_provider("fake-chat", _FakeChat)


def test_vault_target_default(tmp_path):
    t = VaultTarget(root=tmp_path)
    assert t.resolve("MyDoc") == tmp_path / "MyDoc"


def test_vault_target_named(tmp_path):
    t = VaultTarget(root=tmp_path, override="research")
    assert t.resolve("MyDoc") == tmp_path / "research" / "MyDoc"


def test_vault_target_absolute(tmp_path):
    target_dir = tmp_path / "vaults"
    t = VaultTarget(root=tmp_path, override=str(target_dir))
    assert t.resolve("MyDoc") == target_dir / "MyDoc"


def test_process_document_writes_vault_locally():
    _setup()
    d = Path(tempfile.mkdtemp())
    md = d / "Book.md"
    md.write_text(
        "---\ntitle: T\n---\n# PART I\n\n# Introduction\nIntro.\n\n# PART II\n\n"
        "# Chapter One\nBody.\n\n# References\nSmith, 2020.\n"
    )

    vault_root = Path(tempfile.mkdtemp())
    summary = process_document(
        md, api_base=None, vault_target=VaultTarget(root=vault_root),
        chat_provider="fake-chat", ocr_provider="fake-ocr", genre="book",
    )
    assert summary["genre"] == "book"
    out_dir = Path(summary["vault_dir"])
    assert out_dir.exists()
    rels = {p.name for p in out_dir.iterdir()}
    assert "contents.md" in rels
    assert "bibliographic.md" in rels
    assert any(p.name.startswith("0") and p.suffix == ".md" for p in out_dir.iterdir())


def test_process_document_named_vault_override():
    _setup()
    d = Path(tempfile.mkdtemp())
    md = d / "Article.md"
    md.write_text("# Abstract\nA.\n\n# Intro\nB.\n\n# References\n[1] X.\n")

    vault_root = Path(tempfile.mkdtemp())
    summary = process_document(
        md, api_base=None,
        vault_target=VaultTarget(root=vault_root, override="research"),
        chat_provider="fake-chat", ocr_provider="fake-ocr", genre="academic_article",
    )
    assert summary["vault_dir"] == str(vault_root / "research" / "Article")
