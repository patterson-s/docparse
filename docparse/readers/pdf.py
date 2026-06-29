"""PDF → Markdown via Mistral OCR."""

from __future__ import annotations

from pathlib import Path

from mistralai import Mistral


def pdf_to_markdown(path: str | Path, api_key: str) -> str:
    path = Path(path)
    client = Mistral(api_key=api_key)

    with open(path, "rb") as f:
        uploaded = client.files.upload(
            file={"file_name": path.name, "content": f},
            purpose="ocr",
        )

    try:
        signed = client.files.get_signed_url(file_id=uploaded.id, expiry=1)
        result = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "document_url", "document_url": signed.url},
            include_image_base64=False,
            image_limit=0,
        )
    finally:
        try:
            client.files.delete(file_id=uploaded.id)
        except Exception:
            pass

    pages = [page.markdown for page in result.pages]
    return "\n\n---\n\n".join(pages)
