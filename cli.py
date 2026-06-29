"""docparse CLI â€” parse documents into structured JSON."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(help="Parse PDF, DOCX, or MD files into structured chunk JSON.")

_KEY_OPTION = typer.Option(
    None,
    "--api-key",
    envvar="MISTRAL_API_KEY",
    help="Mistral API key (or set MISTRAL_API_KEY env var).",
)


def _require_key(api_key: Optional[str]) -> str:
    if not api_key:
        typer.echo(
            "Error: Mistral API key required. Pass --api-key or set MISTRAL_API_KEY.",
            err=True,
        )
        raise typer.Exit(1)
    return api_key


@app.command()
def parse(
    file: Path = typer.Argument(..., help="PDF, DOCX, or MD file to parse"),
    model: str = typer.Option("mistral-medium-latest", "--model", "-m", help="Mistral model"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output JSON path (default: <stem>_parsed.json)"),
    api_key: Optional[str] = _KEY_OPTION,
) -> None:
    """Parse a single document into chunked JSON."""
    from docparse import parser as _parser

    key = _require_key(api_key)

    if not file.exists():
        typer.echo(f"Error: {file} not found", err=True)
        raise typer.Exit(1)

    if out is None:
        out = file.parent / f"{file.stem}_parsed.json"

    typer.echo(f"Parsing {file.name}...")
    doc = _parser.parse(file, model=model, api_key=key)
    _parser.save(doc, out)

    typer.echo(f"  Source format : {doc.source_format}")
    typer.echo(f"  Sections      : {len(doc.sections)}")
    typer.echo(f"  Chunks        : {len(doc.chunks)}")
    typer.echo(f"  Output        : {out}")


@app.command("parse-dir")
def parse_dir(
    directory: Path = typer.Argument(..., help="Directory containing PDF/DOCX/MD files"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory (default: <dir>/parsed/)"),
    model: str = typer.Option("mistral-medium-latest", "--model", "-m", help="Mistral model"),
    workers: int = typer.Option(1, "--workers", "-w", help="Parallel workers"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Recurse into subdirectories"),
    api_key: Optional[str] = _KEY_OPTION,
) -> None:
    """Parse all PDF/DOCX/MD files in a directory."""
    from concurrent.futures import ThreadPoolExecutor
    from docparse import parser as _parser

    key = _require_key(api_key)

    if not directory.is_dir():
        typer.echo(f"Error: {directory} is not a directory", err=True)
        raise typer.Exit(1)

    if out_dir is None:
        out_dir = directory / "parsed"
    out_dir.mkdir(parents=True, exist_ok=True)

    EXTS = {".pdf", ".docx", ".doc", ".md", ".txt"}
    glob = "**/*" if recursive else "*"
    files = [f for f in directory.glob(glob) if f.is_file() and f.suffix.lower() in EXTS]

    if not files:
        typer.echo(f"No supported files found in {directory}")
        raise typer.Exit(0)

    typer.echo(f"Found {len(files)} file(s) in {directory}")

    def process_one(f: Path) -> None:
        out = out_dir / f"{f.stem}_parsed.json"
        try:
            doc = _parser.parse(f, model=model, api_key=key)
            _parser.save(doc, out)
            typer.echo(f"  âœ“  {f.name}  â†’  {len(doc.chunks)} chunks")
        except Exception as e:
            typer.echo(f"  âœ—  {f.name}: {e}", err=True)

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            executor.map(process_one, files)
    else:
        for f in files:
            process_one(f)

    typer.echo(f"\nDone. Output directory: {out_dir}")


@app.command("parse-structured")
def parse_structured(
    file: Path = typer.Argument(..., help="PDF, DOCX, or MD file to parse"),
    model: str = typer.Option("mistral-medium-latest", "--model", "-m", help="Mistral model"),
    out_dir: Optional[Path] = typer.Option(None, "--out-dir", help="Output directory (default: same as input file)"),
    api_key: Optional[str] = _KEY_OPTION,
) -> None:
    """Structured parse: survey doc type/languages, plan sections, produce language-split outputs.

    Outputs per document:
      {stem}_raw.md          â€” verbatim OCR markdown
      {stem}_structured.md   â€” combined, all sections labeled [EN]/[XHO]/etc.
      {stem}_{lang}.md       â€” one file per detected language
      {stem}_structured.json â€” all chunks with language field
    """
    from docparse.readers import read, detect_format
    from docparse import structurer as _structurer
    from docparse.parser import _to_slug
    import json

    key = _require_key(api_key)

    if not file.exists():
        typer.echo(f"Error: {file} not found", err=True)
        raise typer.Exit(1)

    dest = out_dir or file.parent
    dest.mkdir(parents=True, exist_ok=True)
    stem = file.stem

    typer.echo(f"Parsing {file.name}...")

    # Step 1: OCR / read
    typer.echo("  [1/4] Reading document...")
    raw_markdown = read(file, api_key=key)
    line_count = len(raw_markdown.splitlines())
    (dest / f"{stem}_raw.md").write_text(raw_markdown, encoding="utf-8")
    typer.echo(f"        {line_count} lines  ->  {stem}_raw.md")

    # Step 2: Survey
    typer.echo("  [2/4] Surveying document structure...")
    doc_id = _to_slug(stem)
    doc = _structurer.structure(
        raw_markdown=raw_markdown,
        filename=file.name,
        document_id=doc_id,
        model=model,
        api_key=key,
    )
    p = doc.profile
    typer.echo(f"        {p.doc_type} Â| {', '.join(p.languages)} Â| {p.structure_pattern}")
    typer.echo(f"  [3/4] Structure plan: {len(doc.sections)} sections")

    # Step 3: Write outputs
    typer.echo("  [4/4] Writing output files...")
    written: list[str] = []

    combined_md = _structurer.to_combined_markdown(doc)
    (dest / f"{stem}_structured.md").write_text(combined_md, encoding="utf-8")
    written.append(f"{stem}_structured.md")

    structured_json = _structurer.to_structured_json(doc)
    (dest / f"{stem}_structured.json").write_text(structured_json, encoding="utf-8")
    written.append(f"{stem}_structured.json")

    for lang_slug in _structurer.detected_language_slugs(doc):
        lang_md = _structurer.to_language_markdown(doc, lang_slug)
        if lang_md:
            fname = f"{stem}_{lang_slug}.md"
            (dest / fname).write_text(lang_md, encoding="utf-8")
            written.append(fname)

    for f_name in written:
        typer.echo(f"        ->  {f_name}")
    typer.echo(f"\nDone. Output: {dest}")


@app.command()
def serve(
    port: int = typer.Option(8501, "--port", "-p", help="Streamlit port"),
) -> None:
    """Launch the Streamlit UI."""
    import subprocess

    app_py = Path(__file__).parent / "app.py"
    if not app_py.exists():
        typer.echo(f"Error: {app_py} not found", err=True)
        raise typer.Exit(1)

    cmd = [sys.executable, "-m", "streamlit", "run", str(app_py), "--server.port", str(port)]
    typer.echo(f"Launching Streamlit on port {port}...")
    subprocess.run(cmd)


if __name__ == "__main__":
    app()

