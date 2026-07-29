"""Cost model + structure-quality metrics for the eval harness.

Costs are APPROXIMATE public list prices (USD per 1M tokens) used only to rank
providers in the report. Override via COST_TABLE if prices change. We do not
have exact token counts from every backend, so we estimate from word counts
(words * 1.3 ~= tokens) for OCR text and the prompt size.
"""

from __future__ import annotations

import re

# (input $/1M, output $/1M) — approximate, as of 2024-2025 public list pricing.
# Mistral OCR is priced per page, not tokens; we estimate via page count.
COST_TABLE: dict[str, dict] = {
    "mistral-medium-latest": {"in": 0.27, "out": 0.81},
    "mistral-large-latest": {"in": 2.0, "out": 6.0},
    "mistral-small-latest": {"in": 0.10, "out": 0.30},
    "deepseek-chat": {"in": 0.27, "out": 1.10},
    "qwen-plus": {"in": 0.39, "out": 1.57},
    "qwen-max": {"in": 1.60, "out": 6.40},
}

# Mistral OCR list price: $0.001 per page (estimate). Used to cost the OCR phase.
MISTRAL_OCR_PER_PAGE = 0.001


def _words(text: str) -> int:
    return len(re.findall(r"\S+", text))


def estimate_cost(model: str, prompt_text: str, output_text: str, ocr_pages: int = 0) -> float:
    """Rough USD cost estimate for one chat call (+ optional OCR pages)."""
    table = COST_TABLE.get(model, {"in": 1.0, "out": 3.0})
    in_tok = _words(prompt_text) * 1.3
    out_tok = _words(output_text) * 1.3
    cost = (in_tok / 1_000_000) * table["in"] + (out_tok / 1_000_000) * table["out"]
    cost += ocr_pages * MISTRAL_OCR_PER_PAGE
    return round(cost, 6)


# ── Structure-quality metrics ──────────────────────────────────────────────────

def _tokens(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9']+", text.lower()))


def gold_sections(gold_md: str) -> list[str]:
    """Extract heading labels from a gold-standard markdown (lines starting #)."""
    out = []
    for line in gold_md.splitlines():
        m = re.match(r"^#{1,6}\s+(.*\S)\s*$", line.strip())
        if m:
            out.append(m.group(1).strip())
    return out


def evaluate(gold_md: str, sdoc_sections: list) -> dict:
    """Score a structured result against a gold markdown.

    Metrics:
      - gold_token_f1 : token overlap of full body vs gold body (OCR fidelity)
      - num_gold_sections / num_pred_sections
      - num_unlabeled : gap-filler sections the model failed to name (quality gap)
      - hallucinated   : predicted labels NOT present (as substring) in gold headings
                         nor clearly a gold heading paraphrase (heuristic: not a
                         substring of any gold heading and not matching a gold head
                         after normalization)
      - coverage_ok    : every line in exactly one section (no gaps) — the
                         _fill_gaps invariant guarantees this, so it's a sanity check
    """
    gold_body = re.sub(r"^---.*?---\s*", "", gold_md, flags=re.DOTALL)
    gold_tok = _tokens(gold_body)
    pred_text = "\n\n".join(s.content for s in sdoc_sections)
    pred_tok = _tokens(pred_text)
    inter = gold_tok & pred_tok
    precision = len(inter) / max(len(pred_tok), 1)
    recall = len(inter) / max(len(gold_tok), 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    g_heads = gold_sections(gold_md)
    g_norm = {_norm(h) for h in g_heads}

    pred_labels = [s.label.strip() for s in sdoc_sections if s.label.strip()]
    num_unlabeled = sum(1 for s in sdoc_sections if not s.label.strip())

    hallucinated = []
    for lab in pred_labels:
        n = _norm(lab)
        # A prediction is "real" if it matches a gold heading (or gold contains it).
        if n in g_norm:
            continue
        if any(n in gh or gh in n for gh in g_norm):
            continue
        hallucinated.append(lab)

    return {
        "gold_token_f1": round(f1, 4),
        "num_gold_sections": len(g_heads),
        "num_pred_sections": len(sdoc_sections),
        "num_labeled": len(pred_labels),
        "num_unlabeled": num_unlabeled,
        "hallucinated_sections": hallucinated,
        "num_hallucinated": len(hallucinated),
    }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
