"""Signal adapter for docparse (via signald / signal-cli REST API).

Signal has no native bot API, so this adapter talks to a local `signald`
(or `signal-cli` REST API) daemon over its WebSocket/REST interface, using the
`signalbot` library. The daemon must be running on this PC with your phone
number registered (see DESIGN.md for the Windows/WSL setup).

Run locally:  docparse signal
  - auto-starts the local FastAPI (like the Telegram adapter)
  - connects to signald at SIGNAL_CLI_REST_API (default http://127.0.0.1:8080)
  - processes incoming PDF/DOCX/MD attachments + URLs, writing vaults to a
    local folder.

Vault destination (same model as the Telegram adapter):
  - Default: ~/docparse_vaults/<case>/
  - /vault <name>        -> ~/docparse_vaults/<name>/<case>/
  - /vault /abs/path     -> /abs/path/<case>/
  - /genre <name>        -> force academic_article | book | legal_act

The core `process_document` logic is shared (imported from telegram_bot) — only
the transport/handler layer differs.
"""

from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

from signalbot import Command

from docparse.telegram_bot import VaultTarget, process_document, _process_via_api_url


# ── Pure command parsing (unit-testable, no Signal dependency) ──────────────────

def parse_control_message(text: str) -> tuple[str, Optional[str]]:
    """Interpret a control message.

    Returns (action, arg):
      ("genre", "<name>")  for /genre <name>
      ("vault", "<value>") for /vault <value>  (name or absolute path)
      ("help",  None)      for /help
      ("url",   "<url>")   for an http(s) URL
      ("ignore", None)     for other commands / unknown
      ("doc",   None)      for plain text (shouldn't normally parse docs)
    """
    text = (text or "").strip()
    if not text:
        return ("ignore", None)
    low = text.lower()
    if low.startswith("/genre"):
        parts = text.split(maxsplit=1)
        return ("genre", parts[1].strip() if len(parts) > 1 else "")
    if low.startswith("/vault") or low.startswith("/target"):
        parts = text.split(maxsplit=1)
        return ("vault", parts[1].strip() if len(parts) > 1 else "")
    if low.startswith("/help"):
        return ("help", None)
    if text.startswith("http://") or text.startswith("https://"):
        return ("url", text.strip())
    if low.startswith("/"):
        return ("ignore", None)
    return ("doc", None)


# ── Adapter ─────────────────────────────────────────────────────────────────────

class SignalAdapter:
    def __init__(
        self,
        api_base: str,
        vault_root: Path = VaultTarget().root,
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
        self._chat_state: dict[str, dict] = {}

    def _state(self, chat_id: str) -> dict:
        st = self._chat_state.setdefault(chat_id, {})
        st.setdefault("genre", self.default_genre)
        st.setdefault("vault", None)
        return st

    def help_text(self) -> str:
        named = ", ".join(VaultTarget.list_named(self.vault_root)) or "(none yet)"
        return (
            "docparse (Signal)\n"
            "• Send a PDF / DOCX / MD attachment, or a document URL.\n"
            "• /genre <academic_article|book|legal_act>  (default: auto)\n"
            "• /vault <name or /abs/path>  (default: ~/docparse_vaults)\n"
            f"• Named vaults available: {named}\n"
            "Vaults are written to a local folder on this PC."
        )

    async def handle_message(self, context) -> None:
        """Handle one incoming Signal message (called by DocparseCommand)."""
        msg = context.message
        recipient = msg.recipient if getattr(msg, "recipient", None) else (msg.group or "")
        chat_id = str(recipient)
        st = self._state(chat_id)

        text = getattr(msg, "text", "") or ""
        action, arg = parse_control_message(text)

        if action == "help":
            await context.send(self.help_text())
            return
        if action == "genre":
            st["genre"] = arg or None
            await context.send(f"Genre set to: {st['genre'] or 'auto'}")
            return
        if action == "vault":
            st["vault"] = arg or None
            await context.send(f"Vault target: {st['vault'] or 'default (~/docparse_vaults)'}")
            return
        if action == "url":
            await self._process_url(context, st, arg)
            return
        if action == "ignore":
            await context.send(self.help_text())
            return

        # Document attachment (or attachments).
        attachments = getattr(msg, "attachments_local_filenames", []) or []
        if attachments:
            await self._process_files(context, st, [Path(p) for p in attachments])
            return

        await context.send(
            "Send me a PDF, DOCX, or Markdown file (or a document URL). "
            "Use /genre or /vault to configure, /help for details."
        )

    async def _process_files(self, context, st, paths: list[Path]) -> None:
        await context.send(f"Parsing {len(paths)} file(s)… (this can take ~30s)")
        for path in paths:
            try:
                target = VaultTarget(root=self.vault_root, override=st.get("vault"))
                summary = process_document(
                    path, api_base=self.api_base, vault_target=target,
                    chat_provider=self.chat_provider, ocr_provider=self.ocr_provider,
                    model=self.model, genre=st.get("genre"), api_key=self.api_key,
                )
            except Exception as exc:
                await context.send(f"Error on {path.name}: {exc}")
                continue
            await context.send(
                f"Done — {path.name}\n"
                f"  genre: {summary['genre']}\n"
                f"  vault: {summary['vault_dir']}\n"
                f"  files: {len(summary['files'])}"
            )

    async def _process_url(self, context, st, url: str) -> None:
        await context.send("Parsing URL… (this can take ~30s)")
        try:
            result = _process_via_api_url(
                url, api_base=self.api_base, chat=self.chat_provider,
                ocr=self.ocr_provider, model=self.model, genre=st.get("genre"),
                api_key=self.api_key,
            )
            case = url.rstrip("/").split("/")[-1].split("?")[0] or "url_doc"
            target = VaultTarget(root=self.vault_root, override=st.get("vault"))
            vault_dir = target.resolve(case)
            vault_dir.mkdir(parents=True, exist_ok=True)
            written = []
            for f in result.get("vault_files") or []:
                dest = vault_dir / f["rel_path"]
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(f["content"], encoding="utf-8")
                written.append(f["rel_path"])
        except Exception as exc:
            await context.send(f"Error: {exc}")
            return
        await context.send(
            f"Done — {case}\n  vault: {vault_dir}\n  files: {len(written)}"
        )

    def run(self, signald_url: str = "http://127.0.0.1:8080", phone_number: str = "") -> None:
        from signalbot import SignalBot, Config
        from docparse.signal_bot import DocparseCommand

        # signalbot talks to signald's REST API (default :8080). The phone number
        # must be registered with signald beforehand (see DESIGN.md). Attachments
        # are downloaded so attachments_local_filenames is populated.
        config = Config(
            phone_number=phone_number,
            signal_cli_rest_api=signald_url,
            download_attachments=True,
        )
        bot = SignalBot(config)
        bot.register(DocparseCommand(self))
        bot.start()


class DocparseCommand(Command):
    """signalbot Command wrapper (registered with the bot)."""

    def __init__(self, adapter: SignalAdapter):
        super().__init__()
        self.adapter = adapter

    def setup(self) -> None:
        return

    async def handle(self, context) -> None:
        await self.adapter.handle_message(context)
