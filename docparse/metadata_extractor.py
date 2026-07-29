"""Front-matter extraction via a pluggable ChatProvider (Mistral default).

Refactored to accept a ChatProvider instance so the chat backend is swappable
(e.g. DeepSeek, Qwen) without changing the prompt or call shape. Provider
resolution keeps Mistral as the zero-config default.
"""

from __future__ import annotations

import json

from . import providers

_SYSTEM = """Extract document metadata from the beginning of this document.
Return JSON with these exact keys:
  title     (string or null)
  authors   (list of strings, or empty list)
  year      (integer or null)
  abstract  (string or null)
  source    (journal name, organization, or null)
  doi       (DOI string if visible, e.g. "10.1000/xyz123", or null)

Use null for any field not clearly present. Do not invent information."""


def extract(
    text: str,
    model: str = "mistral-medium-latest",
    api_key: str = "",
    chat_provider=None,
) -> dict:
    """Return a dict with title/authors/year/abstract/source from the first ~2000 chars."""
    if chat_provider is None:
        chat_provider = providers.get_chat_provider(api_key=api_key)
    elif isinstance(chat_provider, str):
        chat_provider = providers.get_chat_provider(chat_provider, api_key=api_key)

    return chat_provider.complete_json(
        _SYSTEM, text[:2000], response_format={"type": "json_object"}, model=model
    )
