"""Front-matter extraction via Mistral (title, authors, year, abstract, source)."""

from __future__ import annotations

import json

_SYSTEM = """Extract document metadata from the beginning of this document.
Return JSON with these exact keys:
  title     (string or null)
  authors   (list of strings, or empty list)
  year      (integer or null)
  abstract  (string or null)
  source    (journal name, organization, or null)

Use null for any field not clearly present. Do not invent information."""

_MAX_RETRIES = 3


def extract(text: str, model: str = "mistral-medium-latest", api_key: str = "") -> dict:
    """Return a dict with title/authors/year/abstract/source from the first ~2000 chars."""
    from mistralai import Mistral
    client = Mistral(api_key=api_key)

    for attempt in range(_MAX_RETRIES):
        try:
            response = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": text[:2000]},
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return json.loads(response.choices[0].message.content)
        except Exception:
            if attempt == _MAX_RETRIES - 1:
                return {}
    return {}
