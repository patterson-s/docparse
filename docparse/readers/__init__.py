"""Format-dispatching reader: PDF → Mistral OCR, DOCX → python-docx, MD → passthrough."""

from __future__ import annotations

from pathlib import Path


def read(path: str | Path, api_key: str = "") -> str:
    """Read a file into a markdown string, dispatching by extension."""
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".pdf":
        from .pdf import pdf_to_markdown
        return pdf_to_markdown(p, api_key=api_key)
    elif ext in {".docx", ".doc"}:
        from .docx_reader import docx_to_markdown
        return docx_to_markdown(p)
    elif ext in {".md", ".txt"}:
        from .md_reader import md_to_markdown
        return md_to_markdown(p)
    else:
        raise ValueError(f"Unsupported file type: {ext!r}. Supported: .pdf, .docx, .doc, .md, .txt")


def detect_format(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    elif ext in {".docx", ".doc"}:
        return "docx"
    elif ext in {".md", ".txt"}:
        return "md"
    return "unknown"
