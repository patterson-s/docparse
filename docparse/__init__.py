"""docparse — standalone document parser for PDFs, DOCX, and Markdown."""

from .models import ParsedDoc, Section, Chunk
from .parser import parse, save

__all__ = ["ParsedDoc", "Section", "Chunk", "parse", "save"]
