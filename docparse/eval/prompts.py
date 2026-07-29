"""Prompt variations for the eval harness.

Each variant is a named pair of (survey_system, plan_system_fn). The `plan_system_fn`
has the same signature as structurer._build_plan_system so it can be passed
straight into structurer.structure(plan_system_fn=...).

The DEFAULT variant reproduces the current production prompt exactly, so the
harness's baseline matches what ships. CONSERVATIVE / STRICT try to fix the
over-segmentation / heading-hallucination we saw on Haas.

Add new variants here and they immediately become available to `docparse eval`.
"""

from __future__ import annotations

from docparse.models import DocProfile
from docparse.structurer import _build_plan_system


def _strict_plan_system(profile: DocProfile, plan_hint: str = "") -> str:
    """Like the default plan prompt but with anti-hallucination guardrails.

    Key additions vs default:
      - Only emit a section if a REAL heading/title line exists in the text.
      - Do NOT invent titles from body prose.
      - Keep the document's own heading levels (e.g. H3 subsections stay H3).
      - Attach unheaded body prose to the preceding section.
    """
    lang_list = ", ".join(f'"{l}"' for l in profile.languages) or '"unknown"'
    hint_block = f"\n\nGENRE-SPECIFIC INSTRUCTION:\n{plan_hint}\n" if plan_hint else ""
    return f"""You are structuring a {profile.doc_type} document.
Languages present: {", ".join(profile.languages)}.
Structure pattern: {profile.structure_pattern}.
Notes: {profile.structure_notes}
{hint_block}
The document text has line numbers prefixed (e.g. "1: text").
Identify the document's REAL sections and map them to exact line ranges.

Rules (strict):
- A section MUST correspond to an actual heading in the text: a markdown heading
  (#, ##, ###), a numbered heading (1. , 2.1 ), or a short isolated title line.
  DO NOT create a section whose label is paraphrased or inferred from body prose.
- If body text appears between two headings, attach it to the PRECEDING section.
  Do not start a new section for unheaded prose.
- Keep the document's own heading levels (a "###" in the text is level 3; do not
  force everything to level 1). Use level 1 for top-level parts/chapters, 2 for
  sections, 3 for subsections.
- Every line must be in exactly one section — no gaps, no overlaps.
- section_id: lowercase slug, e.g. "chapter_1_en", "preamble_xhosa", "toc".
- language: one of {lang_list} or "both" (cover page, TOC). Use short codes
  ("en", "xhosa", "fr") lowercase.
- start_line and end_line are 1-indexed and inclusive.

Return JSON: {{"sections": [{{"label": str, "section_id": str, "language": str, "level": int, "start_line": int, "end_line": int}}, ...]}}"""


def _conservative_plan_system(profile: DocProfile, plan_hint: str = "") -> str:
    """Middle ground: same as default but explicitly forbids title invention."""
    lang_list = ", ".join(f'"{l}"' for l in profile.languages) or '"unknown"'
    hint_block = f"\n\nGENRE-SPECIFIC INSTRUCTION:\n{plan_hint}\n" if plan_hint else ""
    return f"""You are structuring a {profile.doc_type} document.
Languages present: {", ".join(profile.languages)}.
Structure pattern: {profile.structure_pattern}.
Notes: {profile.structure_notes}
{hint_block}
The document text has line numbers prefixed (e.g. "1: text").
Identify ALL top-level sections with their exact line ranges.

Rules:
- Every line must be in exactly one section — no gaps, no overlaps.
- Only label a section from a REAL heading present in the text. If no explicit
  heading precedes a block of body text, do NOT invent a title — the gap-filler
  will assign it an empty label (treated as "unlabeled").
- section_id: lowercase slug, e.g. "chapter_1_en", "preamble_xhosa", "toc".
- language: one of {lang_list} or "both" (for table of contents, cover page, etc.)
  Use short codes: "en", "xhosa", "fr", etc. — match what you see in the languages
  list but lowercase.
- level: 1 for top-level sections, 2 for subsections, 3 for sub-subsections.
- start_line and end_line are 1-indexed and inclusive.

Return JSON: {{"sections": [{{"label": str, "section_id": str, "language": str, "level": int, "start_line": int, "end_line": int}}, ...]}}"""


# Registry: name -> (survey_system_or_None, plan_system_fn_or_None)
# None means "use the production default for that phase".
PROMPT_VARIANTS: dict[str, tuple[str | None, callable | None]] = {
    "default": (None, None),
    "conservative": (None, _conservative_plan_system),
    "strict": (None, _strict_plan_system),
}


def get_variant(name: str) -> tuple[str | None, callable | None]:
    if name not in PROMPT_VARIANTS:
        raise KeyError(f"Unknown prompt variant {name!r}. Known: {list(PROMPT_VARIANTS)}")
    return PROMPT_VARIANTS[name]


def available_variants() -> list[str]:
    return sorted(PROMPT_VARIANTS)
