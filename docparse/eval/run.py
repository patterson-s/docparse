"""Eval runner: execute one (case × provider × prompt-variant) config.

The runner reuses the production pipeline primitives but injects the prompt
variant and measures latency + estimated cost so the harness can compare
providers and prompts on equal footing.
"""

from __future__ import annotations

import time
from pathlib import Path

from docparse.providers import DocumentSource, get_chat_provider, get_ocr_provider
from docparse import structurer
from docparse.genres import route_genre
from docparse.models import StructuredDoc
from docparse.chunker import build_structured_chunks
from docparse.parser import _to_slug
from docparse.eval import metrics


def run_case(
    pdf_path: str | Path,
    gold_md_path: str | Path,
    *,
    chat_provider: str = "mistral",
    ocr_provider: str = "mistral",
    model: str = "mistral-medium-latest",
    api_key: str = "",
    prompt_variant: str = "default",
    genre_override: str | None = None,
    write_outputs: bool = True,
    out_dir: str | Path | None = None,
) -> dict:
    """Run one config on one PDF and return a result dict (no exceptions escape
    except ProviderError, which becomes an 'error' field)."""
    pdf_path = Path(pdf_path)
    gold_md_path = Path(gold_md_path)
    variant_survey, variant_plan_fn = __import__(
        "docparse.eval.prompts", fromlist=["get_variant"]
    ).get_variant(prompt_variant)

    result: dict = {
        "case": pdf_path.stem,
        "chat_provider": chat_provider,
        "ocr_provider": ocr_provider,
        "model": model,
        "prompt_variant": prompt_variant,
        "genre": None,
        "error": None,
        "metrics": {},
        "cost_usd": 0.0,
        "latency_s": 0.0,
        "profile": {},
        "section_labels": [],
    }

    t0 = time.time()
    try:
        ocr = get_ocr_provider(ocr_provider, api_key=api_key)
        chat = get_chat_provider(chat_provider, api_key=api_key)

        src = DocumentSource.from_path(pdf_path)
        raw_markdown = ocr.extract(src)
        ocr_pages = len(raw_markdown.split("\n\n---\n\n")) or raw_markdown.count("---") or 1

        survey_sys = variant_survey  # None -> default
        plan_fn = variant_plan_fn    # None -> default
        if survey_sys is None and plan_fn is None:
            sdoc = structurer.structure(
                raw_markdown, pdf_path.name, _to_slug(pdf_path.stem),
                model=model, chat_provider=chat, api_key=api_key,
            )
        else:
            sdoc = structurer.structure(
                raw_markdown, pdf_path.name, _to_slug(pdf_path.stem),
                model=model, chat_provider=chat, api_key=api_key,
                survey_system=survey_sys, plan_system_fn=plan_fn,
            )

        handler, genre_id = route_genre(sdoc.profile, raw_markdown, override=genre_override)
        result["genre"] = genre_id
        result["profile"] = {
            "doc_type": sdoc.profile.doc_type,
            "languages": sdoc.profile.languages,
            "structure_pattern": sdoc.profile.structure_pattern,
        }
        result["section_labels"] = [s.label or s.section_id for s in sdoc.sections]

        # Metrics
        gold_md = gold_md_path.read_text(encoding="utf-8")
        result["metrics"] = metrics.evaluate(gold_md, sdoc.sections)

        # Cost: OCR pages + the two chat calls (survey sample + full plan).
        plan_prompt = "\n".join(
            [str(survey_sys or ""), raw_markdown[:2000], raw_markdown]
        )  # rough upper bound of what was sent
        cost = metrics.estimate_cost(model, plan_prompt, raw_markdown, ocr_pages=ocr_pages)
        result["cost_usd"] = cost

        # Optional output dump (mirrors what the API would produce).
        if write_outputs:
            out = Path(out_dir) if out_dir else (pdf_path.parent / "eval_out")
            out.mkdir(parents=True, exist_ok=True)
            run_tag = f"{chat_provider}_{prompt_variant}"
            (out / f"{run_tag}_raw.md").write_text(raw_markdown, encoding="utf-8")
            (out / f"{run_tag}_structured.md").write_text(
                structurer.to_combined_markdown(sdoc), encoding="utf-8"
            )

    except Exception as exc:  # surface as error field, don't crash the matrix
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_s"] = round(time.time() - t0, 2)

    return result


def run_matrix(
    cases: list[tuple[str, str]],
    *,
    chat_providers: list[str],
    ocr_provider: str = "mistral",
    models: list[str],
    prompt_variants: list[str],
    api_key: str = "",
    genre_override: str | None = None,
    out_dir: str | Path | None = None,
) -> list[dict]:
    """Run every (case × provider × model × variant) combo. Returns result dicts."""
    results: list[dict] = []
    total = len(cases) * len(chat_providers) * len(models) * len(prompt_variants)
    done = 0
    for pdf, gold in cases:
        for cp in chat_providers:
            for model in models:
                for pv in prompt_variants:
                    done += 1
                    print(f"[{done}/{total}] {Path(pdf).stem} | {cp}/{model} | {pv}")
                    res = run_case(
                        pdf, gold,
                        chat_provider=cp, ocr_provider=ocr_provider, model=model,
                        api_key=api_key, prompt_variant=pv,
                        genre_override=genre_override, out_dir=out_dir,
                    )
                    results.append(res)
    return results
