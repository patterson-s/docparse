"""Eval report: render the matrix results as Markdown + JSON."""

from __future__ import annotations

import json
from pathlib import Path


def write_report(results: list[dict], out_path: str | Path) -> Path:
    """Write <out_path>.md and <out_path>.json. Returns the md path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # JSON
    (out_path.parent / (out_path.stem + ".json")).write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Markdown table
    lines = ["# docparse eval matrix", "", "## Results", ""]
    header = (
        "| case | provider | model | prompt | genre | goldF1 | "
        "#gold | #pred | #unlab | #halluc | cost$ | sec | error |"
    )
    lines.append(header)
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        m = r.get("metrics", {})
        row = (
            f"| {r['case']} | {r['chat_provider']} | {r['model']} | "
            f"{r['prompt_variant']} | {r.get('genre')} | "
            f"{m.get('gold_token_f1', '—')} | {m.get('num_gold_sections', '—')} | "
            f"{m.get('num_pred_sections', '—')} | {m.get('num_unlabeled', '—')} | "
            f"{m.get('num_hallucinated', '—')} | {r.get('cost_usd', 0)} | "
            f"{r.get('latency_s', 0)} | {r.get('error') or '—'} |"
        )
        lines.append(row)

    # Best-config callouts per case (lowest hallucination, then fewest unlabeled).
    lines += ["", "## Per-case best config", ""]
    cases = {}
    for r in results:
        if r.get("error"):
            continue
        cases.setdefault(r["case"], []).append(r)
    for case, rs in cases.items():
        best = min(
            rs,
            key=lambda x: (
                x["metrics"].get("num_hallucinated", 99),
                x["metrics"].get("num_unlabeled", 99),
                -x["metrics"].get("gold_token_f1", 0),
            ),
        )
        m = best["metrics"]
        lines.append(
            f"- **{case}**: `{best['chat_provider']}/{best['model']}/{best['prompt_variant']}` "
            f"— hallucinated={m.get('num_hallucinated')}, unlabeled={m.get('num_unlabeled')}, "
            f"goldF1={m.get('gold_token_f1')}"
        )

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
