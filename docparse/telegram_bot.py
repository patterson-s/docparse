"""Telegram adapter for docparse.

Run locally: `docparse tg`. The bot starts a local FastAPI (uvicorn) in a
background thread, then polls for messages. Send it a PDF/DOCX/MD (or a URL in
a message) and it parses the doc via the local API and writes the vault to a
local folder on this PC.

Vault destination:
  - Default: ~/docparse_vaults/<case>/
  - Per-message override with  /vault <path-or-name>
      * an absolute path is used directly
      * a bare name resolves to ~/docparse_vaults/<name>/<case>/
  - Per-message genre override with  /genre <academic_article|book|legal_act>

Commands are remembered per chat for the next document you send.

The core logic lives in `process_document(...)`, which is provider/transport
agnostic: with `api_base=None` it calls `run_pipeline` directly (used by the
tests, no network); otherwise it POSTs to the local FastAPI.
"""

from __future__ import annotations

import io
import json
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests


# ── Vault target resolution ─────────────────────────────────────────────────────

DEFAULT_VAULT_ROOT = Path.home() / "docparse_vaults"


@dataclass
class VaultTarget:
    """Resolves where a parsed document's vault should be written."""

    root: Path = DEFAULT_VAULT_ROOT
    override: Optional[str] = None  # absolute path, or a named vault

    def resolve(self, case_name: str) -> Path:
        if self.override:
            p = Path(self.override).expanduser()
            if p.is_absolute():
                # An explicit absolute path: treat as the vault *folder* itself
                # if it looks like a case dir, else a parent with case subdir.
                return p / case_name if p.name != case_name else p
            # A bare name -> a named vault under root.
            return self.root / self.override / case_name
        return self.root / case_name

    @staticmethod
    def list_named(root: Path = DEFAULT_VAULT_ROOT) -> list[str]:
        if not root.exists():
            return []
        return sorted(p.name for p in root.iterdir() if p.is_dir())


# ── Core processing (transport-agnostic) ───────────────────────────────────────

def process_document(
    file_path: Path,
    *,
    api_base: Optional[str] = None,
    vault_target: Optional[VaultTarget] = None,
    chat_provider: str = "mistral",
    ocr_provider: str = "mistral",
    model: str = "mistral-medium-latest",
    genre: Optional[str] = None,
    api_key: str = "",
) -> dict:
    """Parse one document and write its vault to disk.

    Returns a summary dict: {case, genre, vault_dir, files, note}.
    """
    vault_target = vault_target or VaultTarget()
    file_path = Path(file_path)
    case = file_path.stem
    vault_dir = vault_target.resolve(case)
    vault_dir.mkdir(parents=True, exist_ok=True)

    if api_base:
        result = _process_via_api(
            file_path, api_base=api_base, chat=chat_provider, ocr=ocr_provider,
            model=model, genre=genre, api_key=api_key,
        )
    else:
        result = _process_direct(
            file_path, chat=chat_provider, ocr=ocr_provider,
            model=model, genre=genre, api_key=api_key, vault_dir=vault_dir,
        )

    genre_id = result.get("genre", genre or "academic_article")

    files = result.get("vault_files") or []
    if files:
        written = []
        for f in files:
            dest = vault_dir / f["rel_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f["content"], encoding="utf-8")
            written.append(f["rel_path"])
    elif result.get("vault_dir") and Path(result["vault_dir"]) == vault_dir and vault_dir.exists():
        # Book path: the pipeline wrote files directly to vault_dir.
        written = sorted(str(p.relative_to(vault_dir)) for p in vault_dir.rglob("*") if p.is_file())
    else:
        written = []

    return {
        "case": case,
        "genre": genre_id,
        "vault_dir": str(vault_dir),
        "files": written,
        "note": result.get("note", ""),
    }


def _process_direct(file_path: Path, *, chat, ocr, model, genre, api_key, vault_dir) -> dict:
    """Direct in-process pipeline (no HTTP). Used by tests and offline runs."""
    from docparse.providers import DocumentSource
    from docparse.pipeline import run_pipeline

    src = DocumentSource.from_path(file_path)
    result = run_pipeline(
        src, filename=file_path.name, chat_provider=chat, ocr_provider=ocr,
        model=model, api_key=api_key, genre_override=genre, return_vault=True,
        vault_dir=vault_dir,
    )
    return result


def _process_via_api(
    file_path: Path, *, api_base, chat, ocr, model, genre, api_key
) -> dict:
    """POST to the local FastAPI, poll the job, download + extract the vault zip."""
    base = api_base.rstrip("/")
    with open(file_path, "rb") as fh:
        files = {"file": (file_path.name, fh, "application/octet-stream")}
        data = {
            "chat_provider": chat,
            "ocr_provider": ocr,
            "model": model,
            "genre": genre or "",
            "api_key": api_key,
        }
        r = requests.post(f"{base}/v1/parse", files=files, data=data, timeout=60)
    if r.status_code != 200:
        raise RuntimeError(f"/v1/parse failed {r.status_code}: {r.text[:200]}")
    job_id = r.json()["job_id"]

    # Poll until done.
    deadline = time.time() + 600
    while time.time() < deadline:
        st = requests.get(f"{base}/v1/jobs/{job_id}", timeout=30).json()
        if st.get("status") == "done":
            break
        if st.get("status") == "error":
            raise RuntimeError(f"job {job_id} errored: {st.get('error')}")
        time.sleep(3)
    else:
        raise RuntimeError(f"job {job_id} timed out")

    # Download vault zip and extract into memory -> return as vault_files.
    zr = requests.get(f"{base}/v1/jobs/{job_id}/download", timeout=60)
    if zr.status_code != 200:
        raise RuntimeError(f"download failed {zr.status_code}")
    vault_files = []
    with zipfile.ZipFile(io.BytesIO(zr.content)) as zf:
        for name in zf.namelist():
            if name.endswith("/"):
                continue
            vault_files.append({"rel_path": name, "content": zf.read(name).decode("utf-8", "replace")})
    return {"genre": genre or "academic_article", "vault_files": vault_files}


# ── Telegram bot ───────────────────────────────────────────────────────────────

class TelegramAdapter:
    def __init__(
        self,
        api_base: str,
        vault_root: Path = DEFAULT_VAULT_ROOT,
        default_genre: Optional[str] = None,
        chat_provider: str = "mistral",
        ocr_provider: str = "mistral",
        model: str = "mistral-medium-latest",
        api_key: str = "",
    ):
        self.api_base = api_base
        self.vault_root = Path(vault_root)
        self.default_genre = default_genre
        self.chat_provider = chat_provider
        self.ocr_provider = ocr_provider
        self.model = model
        self.api_key = api_key
        # Per-chat command state (genre / vault override) remembered between messages.
        self._chat_state: dict[int, dict] = {}

    def _state(self, chat_id: int) -> dict:
        st = self._chat_state.setdefault(chat_id, {})
        st.setdefault("genre", self.default_genre)
        st.setdefault("vault", None)
        return st

    async def handle_message(self, update, context) -> None:
        """Plain-text messages: commands + URLs."""
        text = (update.message.text or "").strip()
        chat_id = update.effective_chat.id
        st = self._state(chat_id)

        if text.startswith("/genre"):
            genre = text.split(maxsplit=1)[1].strip() if " " in text else ""
            st["genre"] = genre or None
            await update.message.reply_text(f"Genre set to: {st['genre'] or 'auto'}")
            return
        if text.startswith("/vault") or text.startswith("/target"):
            vault = text.split(maxsplit=1)[1].strip() if " " in text else ""
            st["vault"] = vault or None
            await update.message.reply_text(f"Vault target: {st['vault'] or 'default (~/docparse_vaults)'}")
            return
        if text.startswith("/help"):
            await update.message.reply_text(self.help_text())
            return
        if text.startswith("/"):
            await update.message.reply_text(self.help_text())
            return

        # A URL -> parse directly.
        if text.startswith("http://") or text.startswith("https://"):
            await self._process(update, st, source_url=text)
            return

        await update.message.reply_text(
            "Send me a PDF, DOCX, or Markdown file (or a document URL). "
            "Use /genre or /vault to configure, /help for details."
        )

    async def handle_document(self, update, context) -> None:
        chat_id = update.effective_chat.id
        st = self._state(chat_id)
        doc = update.message.document
        if not doc:
            return
        # Download the file bytes.
        bot = context.bot
        file_obj = await bot.get_file(doc.file_id)
        bio = io.BytesIO()
        await file_obj.download_to_memory(bio)
        bio.seek(0)
        suffix = Path(doc.file_name or "doc.pdf").suffix or ".pdf"
        tmp = Path.home() / ".cache" / "docparse_tg" / doc.file_name
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_bytes(bio.getvalue())
        await self._process(update, st, file_path=tmp)

    async def _process(self, update, st, file_path=None, source_url=None) -> None:
        await update.message.reply_text("Parsing… (this can take ~30s)")
        try:
            target = VaultTarget(root=self.vault_root, override=st.get("vault"))
            if source_url:
                # For a URL, go through the API (it can OCR the URL directly).
                result = _process_via_api_url(
                    source_url, api_base=self.api_base, chat=self.chat_provider,
                    ocr=self.ocr_provider, model=self.model, genre=st.get("genre"),
                    api_key=self.api_key,
                )
                case = source_url.rstrip("/").split("/")[-1].split("?")[0] or "url_doc"
                vault_dir = target.resolve(case)
                vault_dir.mkdir(parents=True, exist_ok=True)
                written = []
                for f in result.get("vault_files") or []:
                    dest = vault_dir / f["rel_path"]
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_text(f["content"], encoding="utf-8")
                    written.append(f["rel_path"])
                summary = {"case": case, "genre": result.get("genre"),
                           "vault_dir": str(vault_dir), "files": written}
            else:
                summary = process_document(
                    file_path, api_base=self.api_base, vault_target=target,
                    chat_provider=self.chat_provider, ocr_provider=self.ocr_provider,
                    model=self.model, genre=st.get("genre"), api_key=self.api_key,
                )
        except Exception as exc:  # surface errors to the user, don't crash the bot
            await update.message.reply_text(f"Error: {exc}")
            return

        msg = (
            f"Done — genre: {summary['genre']}\n"
            f"Vault: {summary['vault_dir']}\n"
            f"Files: {len(summary['files'])}"
        )
        await update.message.reply_text(msg)

    def help_text(self) -> str:
        named = ", ".join(VaultTarget.list_named(self.vault_root)) or "(none yet)"
        return (
            "docparse bot\n"
            "• Send a PDF / DOCX / MD, or a document URL.\n"
            "• /genre <academic_article|book|legal_act>  (default: auto)\n"
            "• /vault <name or /abs/path>  (default: ~/docparse_vaults)\n"
            f"• Named vaults available: {named}\n"
            "The vault is written to a local folder on this PC."
        )

    def run(self, token: str) -> None:
        from telegram.ext import ApplicationBuilder, MessageHandler, filters

        app = ApplicationBuilder().token(token).build()
        app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_message))
        app.add_handler(MessageHandler(filters.COMMAND, self.handle_message))
        app.run_polling()


def _process_via_api_url(source_url, *, api_base, chat, ocr, model, genre, api_key) -> dict:
    """Parse a URL via the sync endpoint (returns vault_files directly)."""
    base = api_base.rstrip("/")
    r = requests.post(
        f"{base}/v1/parse/sync",
        json={
            "source_url": source_url,
            "chat_provider": chat,
            "ocr_provider": ocr,
            "model": model,
            "genre": genre or "",
            "api_key": api_key,
            "return_vault": True,
        },
        timeout=600,
    )
    if r.status_code != 200:
        raise RuntimeError(f"/v1/parse/sync failed {r.status_code}: {r.text[:200]}")
    data = r.json()
    return {"genre": data.get("genre", genre or "academic_article"),
            "vault_files": data.get("vault_files", [])}
