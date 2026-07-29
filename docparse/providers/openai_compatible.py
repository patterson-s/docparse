"""OpenAI-compatible chat backend (DeepSeek, Qwen, GLM, Kimi, ...).

One implementation covers every vendor that exposes an OpenAI-style
`/chat/completions` endpoint with JSON mode. Wiring a new vendor is just a
registry entry with a `base_url` + env var for the key (see base.py). This is
the "experiment with cheaper CN APIs" entry point.
"""

from __future__ import annotations

import json
import os
from typing import Optional

from .base import ChatProvider, ProviderError

_MAX_RETRIES = 3


class OpenAICompatibleChatProvider(ChatProvider):
    name = "openai-compatible"

    def __init__(
        self,
        base_url: str,
        api_key: str,
        default_model: str = "gpt-4o-mini",
        display_name: str = "openai-compatible",
    ):
        import openai

        self._client = openai.OpenAI(base_url=base_url, api_key=api_key or "EMPTY")
        self._default_model = default_model
        self.display_name = display_name
        self.name = display_name  # reported in metrics / logs

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        response_format: Optional[dict] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ) -> dict:
        for attempt in range(_MAX_RETRIES):
            try:
                kwargs: dict = dict(
                    model=model or self._default_model,
                    messages=messages,
                    temperature=temperature,
                )
                # OpenAI JSON mode uses a string flag; some CN vendors accept
                # {"type": "json_object"} too, but the string form is safest.
                if response_format is not None:
                    rf = response_format.get("type", "json_object")
                    kwargs["response_format"] = {"type": rf}
                response = self._client.chat.completions.create(**kwargs)
                return json.loads(response.choices[0].message.content)
            except Exception:
                if attempt == _MAX_RETRIES - 1:
                    raise ProviderError(
                        f"{self.display_name} chat failed after retries"
                    )
        return {}
