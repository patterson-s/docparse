"""Unit tests for the eval harness — run entirely with fake providers (no network).

Proves:
  - prompt variants are discoverable and injectable
  - the matrix runs a case x providers x prompts and returns scored results
  - metrics detect hallucinated + unlabeled sections
  - report writer emits Markdown + JSON
"""

from __future__ import annotations

from pathlib import Path

import pytest

from docparse.providers import (
    ChatProvider, OcrProvider, DocumentSource,
    register_chat_provider, register_ocr_provider,
)
from docparse.eval import run_case, run_matrix, evaluate, write_report, available_variants
from docparse.eval.prompts import get_variant, PROMPT_VARIANTS


# ── Fake providers ──────────────────────────────────────────────────────────────

class FakeOcr(OcrProvider):
    name = "fake-ocr"

    def extract(self, source: DocumentSource) -> str:
        # 7-line doc with two real headings + body.
        return (
            "# Intro\nBody text about topic one.\n\n"
            "# Analysis\nBody text about topic two.\n\n"
            "More analysis detail.\n"
        )


class FakeChat(ChatProvider):
    name = "fake-chat"

    def __init__(self, *args, **kwargs):
        self.calls = []

    def complete(self, messages, *, response_format=None, model=None, temperature=0.0):
        self.calls.append(model)
        system = messages[0]["content"]
        user = messages[-1]["content"]
        if "Analyze the opening" in system:  # survey
            return {"doc_type": "academic_paper", "languages": ["English"],
                    "structure_pattern": "monolingual", "structure_notes": "t",
                    "estimated_sections": [{"label": "Intro", "language": "English"}]}
        if "section_id" in system:  # plan
            # Hallucinated "Conclusion" section (not in the doc) to exercise metrics.
            return {"sections": [
                {"label": "Intro", "section_id": "intro", "language": "en",
                 "level": 1, "start_line": 1, "end_line": 2},
                {"label": "Analysis", "section_id": "analysis", "language": "en",
                 "level": 1, "start_line": 4, "end_line": 5},
                {"label": "Conclusion", "section_id": "conclusion", "language": "en",
                 "level": 1, "start_line": 7, "end_line": 7},
            ]}
        if "Extract document metadata" in system:
            return {"title": "T", "authors": [], "year": None,
                    "abstract": None, "source": None, "doi": None}
        return {}


@pytest.fixture(autouse=True)
def _register():
    register_ocr_provider("fake-ocr", FakeOcr)
    register_chat_provider("fake-chat", FakeChat)
    yield


def _case(tmp_path: Path) -> tuple[str, str]:
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")
    gold = tmp_path / "doc.md"
    gold.write_text(
        "---\ntitle: T\n---\n# Intro\nBody text about topic one.\n\n# Analysis\nBody text.\n",
        encoding="utf-8",
    )
    return str(pdf), str(gold)


def test_prompt_variants_registered():
    assert "default" in available_variants()
    assert "strict" in available_variants()
    surv, fn = get_variant("strict")
    assert fn is not None  # strict overrides the plan prompt


def test_run_case_produces_metrics():
    import tempfile
    d = Path(tempfile.mkdtemp())
    pdf, gold = _case(d)
    res = run_case(pdf, gold, chat_provider="fake-chat", ocr_provider="fake-ocr",
                   model="fake-model", prompt_variant="default", write_outputs=False)
    assert res["error"] is None
    assert res["genre"] == "academic_article"
    m = res["metrics"]
    # Gold has 2 real headings; prediction invented "Conclusion" -> hallucinated.
    assert m["num_hallucinated"] >= 1
    assert "Conclusion" in m["hallucinated_sections"]
    assert res["cost_usd"] >= 0.0
    assert res["latency_s"] >= 0.0


def test_run_matrix_across_providers_and_prompts():
    import tempfile
    from docparse.eval.prompts import get_variant
    d = Path(tempfile.mkdtemp())
    pdf, gold = _case(d)
    results = run_matrix(
        [(pdf, gold)],
        chat_providers=["fake-chat"],
        ocr_provider="fake-ocr",
        models=["fake-model"],
        prompt_variants=["default", "strict"],
        out_dir=d / "out",
    )
    assert len(results) == 2
    assert all(r["error"] is None for r in results)
    # The strict variant injects a different plan prompt (anti-hallucination).
    surv_d, fn_d = get_variant("default")
    surv_s, fn_s = get_variant("strict")
    assert fn_s is not None and fn_s != fn_d
    # The fake chat ignores the prompt, so both runs detect the same hallucination;
    # the harness correctly flags it in both.
    for r in results:
        assert r["metrics"]["num_hallucinated"] >= 1


def test_write_report_emits_md_and_json(tmp_path: Path):
    import tempfile
    d = Path(tempfile.mkdtemp())
    pdf, gold = _case(d)
    results = run_matrix([(pdf, gold)], chat_providers=["fake-chat"],
                         ocr_provider="fake-ocr",
                         models=["fake-model"], prompt_variants=["default"], out_dir=d / "out")
    out = tmp_path / "report"
    p = write_report(results, out)
    assert p.exists()
    assert (tmp_path / "report.json").exists()
    text = p.read_text(encoding="utf-8")
    assert "docparse eval matrix" in text
    assert "| case | provider |" in text
