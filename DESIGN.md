# docparse — Architecture & Roadmap

docparse turns PDFs/DOCX/MD into structured, genre-aware markdown + an Obsidian
vault. This note describes the **API-only, genre-aware, provider-swappable**
redesign built on top of the original terminal pipeline.

## Principles

1. **Domain logic is transport-agnostic.** `parser`, `structurer`, `chunker`,
   `vault_builder`, `noise_filter`, `metadata_extractor`, `heading_detector`
   operate on *strings and dataclasses*. They never see a terminal, a web
   request, or a filesystem path for the *source*.
2. **Two pluggable seams:** `providers` (who does OCR / chat) and `genres`
   (how the document is understood and laid out). Everything else is shared.
3. **Mistral stays the zero-config default.** Swapping to DeepSeek/Qwen/Paddle
   is a one-word override, never a code fork.

## Layers

```
messaging adapters (Telegram / Signal / Email)   <- future, thin clients
        │  POST /v1/parse {source, genre?, chat?, ocr?}
        ▼
FastAPI gateway (api/)  ── auth, discovery (/v1/genres, /v1/providers)
        │  enqueue
        ▼
docparse/pipeline.run_pipeline(...)   <- the one integration point
        ├─ providers.OcrProvider      (Mistral OCR | Paddle | CN cloud)
        ├─ providers.ChatProvider     (Mistral | DeepSeek | Qwen | GLM | Kimi)
        ├─ genres.GenreRouter         (academic_article | book | legal_act)
        └─ structurer / chunker / vault_builder (shared)
```

## Providers (`docparse/providers/`)

- `ChatProvider` — structured-JSON completion. `MistralChatProvider` and a
  single `OpenAICompatibleChatProvider` base cover DeepSeek/Qwen/GLM/Kimi
  (just a `base_url` + key env var in the registry).
- `OcrProvider` — `DocumentSource` (url OR bytes) → markdown. `MistralOcrProvider`
  uses `document_url` directly when given a URL (no upload round-trip).
- `get_chat_provider(name, api_key)` / `get_ocr_provider(...)` resolve the
  default (Mistral) or a named backend. `register_chat_provider` / `register_ocr_provider`
  add new backends at runtime (used by the experiment harness).

To add a provider: implement the interface, register it. No pipeline edits.

## Genres (`docparse/genres/`)

A genre is a bundle of:
- `confidence(profile, sample)` — routing score (used only without an override)
- `GenreConfig` — `chunk_window`, `chunk_overlap`, `extra_discard_patterns`,
  `plan_hint` (appended to the structurer's plan prompt)
- `build_vault(sdoc, out_dir, metadata, serper)` — the genre-specific layout

| Genre | Behavior |
|---|---|
| `academic_article` (default) | abstract / body / references / bibliographic.md |
| `book` | split into chapters, smaller chunks, discard publisher/copyright/TOC/index front+back matter, per-chapter files + contents.md |
| `legal_act` (stub) | parts→articles→schedules layout deferred; currently flat per-section |

To add a genre: subclass `GenreHandler`, set `id`/`label`/`config`, implement
`build_vault`, register via `genres.register_genre(...)` (or drop into the package).
`GenreRouter` picks the best handler, or honours an explicit `?genre=` override.

## API surface (`docparse/api/`)

- `GET  /health`
- `GET  /v1/genres`  ·  `GET  /v1/providers`
- `POST /v1/parse`        → 202 `{job_id}` (async, BackgroundTasks)
- `POST /v1/parse/sync`   → 200 (full result; small docs/tests only)
- `GET  /v1/jobs/{id}`    → status + result/error
- `GET  /v1/jobs/{id}/download` → vault zip

Auth (bearer) + durable queue (Redis/ARQ/Cloud Tasks) + object storage for the
vault are the next hardening steps before production.

## What's proven

`tests/test_pipeline_providers_genres.py` runs the **entire pipeline** with
FAKE providers + a fake genre (no network, no Mistral key) and asserts:
genre routing, provider swap, vault layout. Run with:

```
python -m pytest tests/ -v
```

## Eval harness (`docparse/eval/`)

Run a **matrix** of (case × chat-provider × model × prompt-variant) and score
each run. Built so prompt and provider experiments are data-driven, not forks.

- `prompts.py` — `PROMPT_VARIANTS` registry: name → (survey_system, plan_fn).
  `default` reproduces the production prompt byte-for-byte; `conservative` /
  `strict` are anti-hallucination variants. Add a variant here and it's
  immediately available to the CLI.
- `metrics.py` — `evaluate(gold_md, sections)` returns: `gold_token_f1`
  (OCR/structure fidelity vs gold body), `num_gold/pred_sections`,
  `num_unlabeled` (gap-filler blocks the model failed to name), `hallucinated`
  (predicted labels not present in the gold headings), plus `estimate_cost`
  (approximate USD from word counts + a price table).
- `run.py` — `run_case(...)` (OCR → survey → plan with variant → genre-aware
  vault; records latency + cost) and `run_matrix(...)` (cartesian product).
- `report.py` — writes `<out>.md` (comparison table + per-case best config)
  and `<out>.json`.

`structurer.structure()` accepts optional `survey_system` / `plan_system_fn`
overrides (default `None` = production prompt), which is how variants inject
without changing the pipeline.

CLI:
```
docparse eval <corpus-or-case-dir> \
  --chat mistral,deepseek --model mistral-medium-latest,deepseek-chat \
  --prompts default,strict --out report
```
A "case" is a folder with one `.pdf` and one gold-standard `.md`.

Run with:
```bash
export $(grep -v '^#' .env | xargs)   # load MISTRAL_API_KEY
python -m cli eval tests/Haas_1992_IntroductionEpistemicCommunities \
  --chat mistral --model mistral-medium-latest --prompts default,conservative,strict
```

Harness tests (no network): `tests/test_eval_harness.py` (4 tests, fake
providers). Full suite: `python -m pytest tests/ -v`.

## Book genre (`docparse/genres/book.py`)

Whole-book → per-chapter Obsidian vault. Books are split **deterministically**
(never by the LLM) because one-shot LLM planning of a 2,000+ line book is
unstable — and we proved on the article matrix that "stricter" prompts
hallucinate 40–50 sections. `split_book(raw_markdown)` is a pure function:

- Strips a leading YAML front-matter block.
- Skips the Contents / Table-of-Contents TOC block (TOC `Chapter N. Title 123`
  lines are detected and dropped so they aren't mistaken for chapters).
- Segments into PARTs + CHAPTERs using `# PART N` / `## CHAPTER N` markers and
  top-level (`#`) headings (a `#` title right after a PART/CHAPTER marker is
  that chapter's own title, not a new chapter; level-2 `##` headings are
  subsections and never break a chapter).
- Separates front matter (Foreword/Acknowledgments/Preface) and back matter
  (References/Notes/Bibliography/Index) — curated into `00_front_matter.md` /
  `zz_back_matter.md`, never polluting the chapter sequence.

`process_book(...)` emits the vault: `bibliographic.md`, `contents.md`
(part+chapter index), `NN_<slug>.md` per chapter, front/back matter, and
`raw.md`. Optional `substructure=True` re-runs the structurer per chapter (with
a chat provider) to add subsections.

The pipeline branches on genre: `genre_id == "book"` calls `process_book`
directly (no wasted generic LLM plan); articles use the normal structurer.

Vault layout for a book:
```
bibliographic.md
contents.md            # reading order (parts + chapters)
00_front_matter.md
01_introduction.md
02_competing_for_authority.md
...
zz_back_matter.md
raw.md
```

Verified on `tests/Sending_2015_PoliticsExpertiseCompeting` (a 420 KB OCR dump,
no gold): produced 3 parts, 7 sections (Introduction + 5 chapters + Conclusion),
with front/back matter correctly separated. Tests: `tests/test_book_genre.py`.

## Two modes for "book chapters as docs"

- **Whole book** → `book` genre (above): splits into chapters.
- **A single chapter already saved as its own file** → use `academic_article`
  (or `book` with a one-chapter input): it's just a doc, structured normally.
  So chapter-level work needs no special path — point the parser at the chapter
  file and let the default genre handle it.

## Telegram adapter (`docparse/telegram_bot.py` + `docparse tg`)

A local-first messaging adapter: run `docparse tg` on your PC and message the
bot a PDF/DOCX/MD (or a URL). It talks to the **local FastAPI** (`/v1/parse`),
then writes the resulting Obsidian vault to a **local folder**.

```bash
export $(grep -v '^#' .env | xargs)            # load MISTRAL_API_KEY + TELEGRAM_BOT_TOKEN
python -m cli tg                                # auto-starts local API, then polls
```

Per-message commands (remembered per chat until changed):
- `/genre <academic_article|book|legal_act>` — override auto-detection.
- `/vault <name or /abs/path>` — where to write. A bare name →
  `~/docparse_vaults/<name>/<case>/`; an absolute path is used directly.
- `/help` — summary.

Default vault root: `~/docparse_vaults/<case>/`. Named vaults list is shown in
`/help`. The core logic (`process_document`, `VaultTarget`) is transport-agnostic
and unit-tested with fake providers (`tests/test_telegram_adapter.py`). With
`api_base=None` it calls `run_pipeline` directly (offline/test mode); otherwise
it POSTs to the FastAPI and extracts the returned vault zip.

## Roadmap / open seams

1. ~~Messaging adapters (Telegram first)~~ — DONE. Next: Signal (needs signald)
   or a multi-tenant gateway.
2. More genres (legal_act full layout, report) and more providers (PaddleOCR,
   Tencent/Qwen OCR, Qwen-VL vision→markdown). To compare CN chat providers,
   add keys (DEEPSEEK_API_KEY / QWEN_API_KEY) and pass `--chat deepseek,qwen`.
3. Durable jobs + per-tenant auth + vault in object storage / git-committed repo.
4. Richer eval signal: LLM-judge quality score vs gold, and a coverage metric
   that checks every gold heading is present in the prediction (not just token F1).
5. Book substructuring: flip `substructure=True` in the pipeline's book branch
   (per-chapter LLM subsectioning) once a chapter-gold corpus exists to eval it.
