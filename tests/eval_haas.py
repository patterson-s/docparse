"""Evaluate the Mistral pipeline against the Haas gold standard.

Runs the full docparse pipeline (Mistral OCR + Mistral chat, academic_article
genre) on the Haas PDF, writes outputs next to the gold standard, and reports:
  - OCR fidelity: token overlap vs the gold body
  - Section detection: labels the structurer produced
  - Vault layout: files produced by the genre handler
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure the project root (containing the `docparse` package) is importable
# regardless of the current working directory.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv()  # reads .env -> MISTRAL_API_KEY

from docparse.providers import DocumentSource, get_chat_provider, get_ocr_provider
from docparse import structurer
from docparse.genres import route_genre
from docparse.models import StructuredDoc
from docparse.chunker import build_structured_chunks
from docparse.parser import _to_slug

CASE = Path(__file__).resolve().parent / "Haas_1992_IntroductionEpistemicCommunities"
PDF = CASE / "Haas - Introduction epistemic communities and international policy coordination.pdf"
GOLD = CASE / "Haas_1992_IntroductionEpistemicCommunities.md"
OUT = CASE / "out"
OUT.mkdir(parents=True, exist_ok=True)

API_KEY = os.environ.get("MISTRAL_API_KEY", "")


def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def _make_sdoc(doc_id, filename, profile, sections, raw_markdown):
    chunks = build_structured_chunks(doc_id, sections, raw_markdown)
    return StructuredDoc(
        document_id=doc_id, filename=filename, profile=profile,
        sections=sections, chunks=chunks, raw_markdown=raw_markdown,
        parsed_at=datetime.now(timezone.utc).isoformat(),
    )


def main():
    if not PDF.exists():
        print(f"MISSING PDF: {PDF}")
        sys.exit(1)
    if not API_KEY:
        print("MISSING MISTRAL_API_KEY in .env")
        sys.exit(1)

    print("=" * 70)
    print("docparse Mistral pipeline — Haas 1992 evaluation")
    print("=" * 70)

    src = DocumentSource.from_path(PDF)

    print(f"[1/4] OCR via Mistral  (file: {PDF.name}, {PDF.stat().st_size // 1024} KB)")
    ocr = get_ocr_provider("mistral", api_key=API_KEY)
    raw_markdown = ocr.extract(src)
    (OUT / "raw.md").write_text(raw_markdown, encoding="utf-8")
    print(f"      OCR produced {len(raw_markdown.splitlines())} lines, "
          f"{len(raw_markdown.split())} words")

    print("[2/4] Survey + structure plan via Mistral chat")
    chat = get_chat_provider("mistral", api_key=API_KEY)
    lines = raw_markdown.splitlines()
    profile = structurer._survey(chat, "mistral-medium-latest", lines)
    print(f"      profile: doc_type={profile.doc_type!r} "
          f"langs={profile.languages} pattern={profile.structure_pattern!r}")
    handler, genre_id = route_genre(profile, raw_markdown, override=None)
    plan = structurer._plan(chat, "mistral-medium-latest", lines, profile,
                            plan_hint=handler.config.plan_hint)
    sections = structurer._execute(lines, plan)
    print(f"      genre routed -> {genre_id}  ({len(sections)} sections)")

    doc_id = _to_slug(PDF.stem)
    sdoc = _make_sdoc(doc_id, PDF.name, profile, sections, raw_markdown)
    combined = structurer.to_combined_markdown(sdoc)
    (OUT / "structured.md").write_text(combined, encoding="utf-8")

    print("[3/4] Build academic-article vault")
    handler.build_vault(
        sdoc,
        OUT / "vault",
        metadata={"title": "Introduction: epistemic communities and international policy coordination",
                  "authors": ["Peter M. Haas"], "year": 1992, "source": "International Organization",
                  "doi": "10.1017/S0020818300001442"},
        serper={},
    )
    print("      vault files:",
          sorted(p.name for p in (OUT / "vault").iterdir() if p.is_file()))

    print("[4/4] Compare OCR vs gold standard")
    gold_text = GOLD.read_text(encoding="utf-8")
    gold_body = re.sub(r"^---.*?---\s*", "", gold_text, flags=re.DOTALL)
    gold_tok = _tokens(gold_body)
    ocr_tok = _tokens(raw_markdown)
    inter = gold_tok & ocr_tok
    precision = len(inter) / max(len(ocr_tok), 1)
    recall = len(inter) / max(len(gold_tok), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    print(f"      gold body tokens: {len(gold_tok)}")
    print(f"      ocr  tokens:      {len(ocr_tok)}")
    print(f"      token overlap precision={precision:.3f} recall={recall:.3f} f1={f1:.3f}")
    print()
    print("      Detected section labels:")
    for s in sections:
        print(f"        [{s.language}] L{s.level}  {s.label or s.section_id}  "
              f"({len(s.content.split())} words)")

    print("\nOutputs written to:", OUT)
    print("Done.")


if __name__ == "__main__":
    main()
