"""Pluggable provider layer.

Two provider families:

  ChatProvider - structured JSON completion (survey / structure plan /
                 metadata / heading detection). OpenAI-compatible backends
                 (DeepSeek, Qwen, GLM, Kimi...) share one implementation.
  OcrProvider  - document -> markdown. Mistral OCR is the default; Paddle
                 (local) and cloud CN OCR backends can be registered later.

The registry functions (get_chat_provider / get_ocr_provider) keep Mistral as
the zero-config default, so the rest of docparse is unchanged when no provider
is passed. Swapping a provider is then a one-word override at the API boundary.
"""

from __future__ import annotations

import os
from typing import Optional

from .base import (
    ChatProvider,
    OcrProvider,
    DocumentSource,
    ProviderError,
)
from .mistral import MistralChatProvider, MistralOcrProvider
from .openai_compatible import OpenAICompatibleChatProvider


__all__ = [
    "ChatProvider",
    "OcrProvider",
    "DocumentSource",
    "ProviderError",
    "get_chat_provider",
    "get_ocr_provider",
    "list_chat_providers",
    "list_ocr_providers",
    "register_chat_provider",
    "register_ocr_provider",
    "MistralChatProvider",
    "MistralOcrProvider",
    "OpenAICompatibleChatProvider",
]


# ── Registry ─────────────────────────────────────────────────────────────────
# Registered lazily to avoid importing vendored SDKs until a provider is used.

_CHAT_REGISTRY: dict[str, tuple[type, dict]] = {
    "mistral": (MistralChatProvider, {}),
    "deepseek": (
        OpenAICompatibleChatProvider,
        {
            "base_url": os.environ.get(
                "DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"
            ),
            "env_key": "DEEPSEEK_API_KEY",
            "default_model": "deepseek-chat",
        },
    ),
    "qwen": (
        OpenAICompatibleChatProvider,
        {
            "base_url": os.environ.get(
                "QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"
            ),
            "env_key": "QWEN_API_KEY",
            "default_model": "qwen-plus",
        },
    ),
}

_OCR_REGISTRY: dict[str, tuple[type, dict]] = {
    "mistral": (MistralOcrProvider, {}),
}


def register_chat_provider(name: str, cls: type, cfg: dict | None = None) -> None:
    """Register a new chat provider at runtime (used by experiments/plugins)."""
    _CHAT_REGISTRY[name] = (cls, cfg or {})


def register_ocr_provider(name: str, cls: type, cfg: dict | None = None) -> None:
    _OCR_REGISTRY[name] = (cls, cfg or {})


def list_chat_providers() -> list[str]:
    return sorted(_CHAT_REGISTRY)


def list_ocr_providers() -> list[str]:
    return sorted(_OCR_REGISTRY)


def get_chat_provider(name: str = "mistral", api_key: Optional[str] = None) -> ChatProvider:
    if name not in _CHAT_REGISTRY:
        raise ProviderError(
            f"Unknown chat provider {name!r}. Known: {list_chat_providers()}"
        )
    cls, cfg = _CHAT_REGISTRY[name]
    if name == "mistral":
        key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY", "")
        return cls(api_key=key)
    # Custom-registered provider (cfg empty): instantiate directly.
    if not cfg:
        key = api_key if api_key is not None else ""
        try:
            return cls(api_key=key)
        except TypeError:
            return cls()
    # OpenAI-compatible backend: needs base_url/key/default_model.
    key = api_key if api_key is not None else os.environ.get(cfg["env_key"], "")
    return cls(
        base_url=cfg["base_url"],
        api_key=key,
        default_model=cfg["default_model"],
        display_name=name,
    )


def get_ocr_provider(name: str = "mistral", api_key: Optional[str] = None) -> OcrProvider:
    if name not in _OCR_REGISTRY:
        raise ProviderError(
            f"Unknown OCR provider {name!r}. Known: {list_ocr_providers()}"
        )
    cls, cfg = _OCR_REGISTRY[name]
    if name == "mistral":
        key = api_key if api_key is not None else os.environ.get("MISTRAL_API_KEY", "")
        return cls(api_key=key)
    # Custom-registered provider (cfg empty): instantiate directly.
    if not cfg:
        key = api_key if api_key is not None else ""
        try:
            return cls(api_key=key)
        except TypeError:
            return cls()
    raise ProviderError(f"OCR provider {name!r} not configured")
