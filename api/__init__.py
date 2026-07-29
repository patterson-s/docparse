"""FastAPI service for docparse.

Endpoints:
  GET  /health                     -> { ok: true }
  POST /v1/parse                   -> 202 { job_id, status }   (async, default)
  POST /v1/parse/sync              -> 200 { full result JSON }  (small docs only)
  GET  /v1/jobs/{job_id}           -> { status, step, total_steps, progress, result|error }
  GET  /v1/jobs/{job_id}/download  -> vault zip (when return_vault)
  GET  /v1/genres                  -> [ "academic_article", "book", "legal_act" ]
  GET  /v1/providers               -> { chat: [...], ocr: [...] }

The service is transport-agnostic about the source: accept either an uploaded
file (multipart) or a `source_url`. A messaging adapter (Telegram/Signal) is a
thin client that posts the same request. Genre + provider are one-word overrides.
"""

from __future__ import annotations

import asyncio
import io
import os
import tempfile
import uuid
import zipfile
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from docparse import pipeline
from docparse.providers import DocumentSource, list_chat_providers, list_ocr_providers
from docparse import genres as genres_mod

app = FastAPI(title="docparse API", version="0.1.0")

# In-memory job store. Swap for Redis/DB in a real deployment.
_JOBS: dict[str, dict] = {}
_LOCK = asyncio.Lock()


# ── Request models ────────────────────────────────────────────────────────────

class ParseRequest(BaseModel):
    source_url: Optional[str] = Field(None, description="URL to OCR directly (preferred)")
    mode: str = Field("structured", description="structured | standard (reserved)")
    model: str = "mistral-medium-latest"
    chat_provider: str = "mistral"
    ocr_provider: str = "mistral"
    api_key: Optional[str] = None  # server-side; usually injected from env
    genre: Optional[str] = None   # override: academic_article | book | legal_act
    return_vault: bool = False
    serper_key: Optional[str] = None


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_api_key(req_key: Optional[str]) -> str:
    return req_key or os.environ.get("MISTRAL_API_KEY", "")


def _build_source(req: ParseRequest, upload: Optional[UploadFile]) -> DocumentSource:
    if req.source_url:
        return DocumentSource.from_url(req.source_url)
    if upload is not None:
        content = upload.file.read()
        return DocumentSource.from_bytes(content, upload.filename or "upload.pdf")
    raise HTTPException(400, "Provide either source_url or an uploaded file")


# ── Health / discovery ────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/v1/genres")
async def list_genres():
    return genres_mod.available_genres()


@app.get("/v1/providers")
async def list_providers():
    return {"chat": list_chat_providers(), "ocr": list_ocr_providers()}


# ── Parse (async via BackgroundTasks) ─────────────────────────────────────────

@app.post("/v1/parse")
async def parse_async(
    background: BackgroundTasks,
    req: ParseRequest = None,
    file: Optional[UploadFile] = File(None),
):
    # FastAPI can't mix a JSON body and multipart easily; accept source_url via
    # form fields when a file is uploaded. For pure-JSON, use the sync endpoint.
    if req is None:
        raise HTTPException(400, "JSON body required")
    job_id = f"j_{uuid.uuid4().hex[:12]}"
    _JOBS[job_id] = {
        "status": "queued",
        "step": 0,
        "total_steps": 4,
        "progress": 0.0,
        "result": None,
        "error": None,
    }
    background.add_task(_run_job, job_id, req, file)
    return {"job_id": job_id, "status": "queued"}


@app.post("/v1/parse/sync")
async def parse_sync(req: ParseRequest, file: Optional[UploadFile] = File(None)):
    """Synchronous parse — only for small docs / tests. Returns full result."""
    source = _build_source(req, file)
    api_key = _resolve_api_key(req.api_key)
    try:
        result = pipeline.run_pipeline(
            source,
            model=req.model,
            api_key=api_key,
            chat_provider=req.chat_provider,
            ocr_provider=req.ocr_provider,
            genre_override=req.genre,
            return_vault=req.return_vault,
            serper_key=req.serper_key or "",
        )
    except Exception as exc:  # surface provider errors to the caller
        raise HTTPException(502, f"Pipeline failed: {exc}")
    return result


# ── Job status / download ──────────────────────────────────────────────────────

@app.get("/v1/jobs/{job_id}")
async def job_status(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    return job


@app.get("/v1/jobs/{job_id}/download")
async def job_download(job_id: str):
    job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "Unknown job id")
    vault_dir = job.get("vault_dir")
    if not vault_dir or not Path(vault_dir).exists():
        raise HTTPException(404, "No vault output for this job")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in Path(vault_dir).rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(vault_dir))
    buf.seek(0)
    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={job_id}.zip"},
    )


# ── Worker ──────────────────────────────────────────────────────────────────────

async def _run_job(job_id: str, req: ParseRequest, upload: Optional[UploadFile]):
    job = _JOBS[job_id]
    try:
        source = _build_source(req, upload)
        api_key = _resolve_api_key(req.api_key)

        job.update(status="ocr", step=1, progress=0.25)
        result = await asyncio.to_thread(
            pipeline.run_pipeline,
            source,
            model=req.model,
            api_key=api_key,
            chat_provider=req.chat_provider,
            ocr_provider=req.ocr_provider,
            genre_override=req.genre,
            return_vault=req.return_vault,
            serper_key=req.serper_key or "",
        )
        if req.return_vault and result.get("vault_dir"):
            job["vault_dir"] = result.pop("vault_dir")
        job.update(status="done", step=4, progress=1.0, result=result)
    except Exception as exc:
        job.update(status="error", error=str(exc))
