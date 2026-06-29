"""DOCX → Markdown via python-docx."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.text.paragraph import Paragraph
from docx.table import Table


def docx_to_markdown(path: str | Path) -> str:
    doc = Document(str(path))
    lines: list[str] = []

    for block in _iter_blocks(doc):
        if isinstance(block, Paragraph):
            text = block.text.strip()
            if not text:
                lines.append("")
                continue
            style = block.style.name if block.style else ""
            if "Heading 1" in style:
                lines.append(f"# {text}")
            elif "Heading 2" in style:
                lines.append(f"## {text}")
            elif "Heading 3" in style:
                lines.append(f"### {text}")
            elif "Heading 4" in style or "Heading 5" in style or "Heading 6" in style:
                lines.append(f"#### {text}")
            else:
                lines.append(text)

        elif isinstance(block, Table):
            table_lines = _table_to_markdown(block)
            if table_lines:
                lines.append("")
                lines.extend(table_lines)
                lines.append("")

    return "\n".join(lines)


def _iter_blocks(doc: Document):
    """Yield paragraphs and tables in document order."""
    for child in doc.element.body:
        tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
        if tag == "p":
            yield Paragraph(child, doc)
        elif tag == "tbl":
            yield Table(child, doc)


def _table_to_markdown(table: Table) -> list[str]:
    rows = []
    for i, row in enumerate(table.rows):
        cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
        rows.append(" | ".join(cells))
        if i == 0:
            rows.append(" | ".join("---" for _ in cells))
    return rows
