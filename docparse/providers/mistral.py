"""Mistral providers: OCR (mistral-ocr-latest) + chat (structured JSON).

Mirrors the exact behaviour previously inline in readers/pdf.py and the
extractor/detector modules, but behind the OcrProvider / ChatProvider
interfaces so it is one option among many.
"""

from __future__ import annotations

import json
from typing import Optional

from .base import ChatProvider, OcrProvider, DocumentSource, ProviderError

_MAX_RETRIES = 3

# Models used when the caller does not specify one.
_OCR_MODEL = "mistral-ocr-latest"
_CHAT_MODEL = "mistral-medium-latest"


class MistralOcrProvider(OcrProvider):
    name = "mistral"

    def __init__(self, api_key: str = ""):
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)

    def extract(self, source: DocumentSource) -> str:
        client = self._client
        file_id: Optional[str] = None
        try:
            if source.url:
                result = client.ocr.process(
                    model=_OCR_MODEL,
                    document={"type": "document_url", "document_url": source.url},
                    include_image_base64=False,
                    image_limit=0,
                )
            elif source.content is not None:
                uploaded = client.files.upload(
                    file={"file_name": source.filename, "content": source.content},
                    purpose="ocr",
                )
                file_id = uploaded.id
                signed = client.files.get_signed_url(file_id=file_id, expiry=1)
                result = client.ocr.process(
                    model=_OCR_MODEL,
                    document={"type": "document_url", "document_url": signed.url},
                    include_image_base64=False,
                    image_limit=0,
                )
            else:
                raise ProviderError("DocumentSource has neither url nor content")

            pages = [page.markdown for page in result.pages]
            return "\n\n---\n\n".join(pages)
        finally:
            if file_id:
                try:
                    client.files.delete(file_id=file_id)
                except Exception:
                    pass


class MistralChatProvider(ChatProvider):
    name = "mistral"

    def __init__(self, api_key: str = ""):
        from mistralai import Mistral

        self._client = Mistral(api_key=api_key)

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
                    model=model or _CHAT_MODEL,
                    messages=messages,
                    temperature=temperature,
                )
                if response_format is not None:
                    kwargs["response_format"] = response_format
                response = self._client.chat.complete(**kwargs)
                return json.loads(response.choices[0].message.content)
            except Exception:
                if attempt == _MAX_RETRIES - 1:
                    raise ProviderError("Mistral chat failed after retries")
        return {}
