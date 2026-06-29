"""Streamlit UI for docparse."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="docparse",
    page_icon="📄",
    layout="wide",
)

# ── Language badge colours ────────────────────────────────────────────────────
_LANG_COLOURS = {
    "en": "#1f77b4", "xhosa": "#ff7f0e", "fr": "#2ca02c",
    "af": "#d62728", "zu": "#9467bd", "both": "#7f7f7f", "unknown": "#bcbcbc",
}

def _lang_badge(lang: str) -> str:
    from docparse.structurer import _lang_slug, _lang_display
    slug = _lang_slug(lang) if lang not in ("both", "unknown", "") else lang
    colour = _LANG_COLOURS.get(slug, "#aaaaaa")
    label = _lang_display(lang) if lang not in ("both", "unknown", "") else lang.upper()
    return f'<span style="background:{colour};color:white;padding:2px 7px;border-radius:4px;font-size:0.75em;font-weight:600">{label}</span>'


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 docparse")
    st.caption("PDF · DOCX · MD → structured JSON")

    api_key = st.text_input(
        "Mistral API key",
        type="password",
        placeholder="your-api-key-here",
    )

    st.divider()

    parse_mode = st.radio(
        "Parse mode",
        ["Standard", "Structured"],
        help="Standard: generic heading detection.  Structured: agentic survey → language-aware section plan.",
    )

    input_method = st.radio("Input", ["Upload file", "Local path"], horizontal=True)

    uploaded_file = None
    local_path = ""

    if input_method == "Upload file":
        uploaded_file = st.file_uploader(
            "Choose a file",
            type=["pdf", "docx", "doc", "md", "txt"],
            label_visibility="collapsed",
        )
    else:
        local_path = st.text_input("File path", placeholder="C:\\path\\to\\document.pdf")

    model = st.selectbox(
        "Model",
        ["mistral-medium-latest", "mistral-large-latest", "mistral-small-latest"],
    )

    parse_btn = st.button("Parse", type="primary", use_container_width=True)


# ── Parse ─────────────────────────────────────────────────────────────────────

def _get_tmp_path(uploaded) -> str:
    suffix = Path(uploaded.name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.read())
        return tmp.name


if parse_btn:
    if not api_key:
        st.error("Enter your Mistral API key in the sidebar.")
    elif not uploaded_file and not local_path.strip():
        st.error("Provide a file upload or local path.")
    else:
        with st.spinner("Parsing document…"):
            try:
                src = _get_tmp_path(uploaded_file) if uploaded_file else local_path.strip()
                fname = uploaded_file.name if uploaded_file else Path(local_path).name

                if parse_mode == "Structured":
                    from docparse import structurer as _structurer
                    from docparse.readers import read
                    from docparse.parser import _to_slug

                    raw_markdown = read(src, api_key=api_key)
                    doc_id = _to_slug(Path(fname).stem)
                    sdoc = _structurer.structure(
                        raw_markdown=raw_markdown,
                        filename=fname,
                        document_id=doc_id,
                        model=model,
                        api_key=api_key,
                    )
                    st.session_state["mode"] = "structured"
                    st.session_state["sdoc"] = sdoc
                    st.session_state["sdoc_dict"] = dataclasses.asdict(sdoc)
                    st.session_state.pop("doc", None)
                else:
                    from docparse import parser as _parser
                    doc = _parser.parse(src, model=model, api_key=api_key)
                    if uploaded_file:
                        doc = dataclasses.replace(doc, filename=fname)
                    st.session_state["mode"] = "standard"
                    st.session_state["doc"] = doc
                    st.session_state["doc_dict"] = dataclasses.asdict(doc)
                    st.session_state.pop("sdoc", None)

            except Exception as e:
                st.error(f"Parse failed: {e}")


# ── Standard Results ──────────────────────────────────────────────────────────

if st.session_state.get("mode") == "standard" and "doc" in st.session_state:
    from docparse.models import ParsedDoc
    doc: ParsedDoc = st.session_state["doc"]
    doc_dict: dict = st.session_state["doc_dict"]

    tab_overview, tab_sections, tab_chunks, tab_export = st.tabs(
        ["Overview", "Sections", "Chunks", "Export"]
    )

    with tab_overview:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Format", doc.source_format.upper())
        col2.metric("Sections", len(doc.sections))
        col3.metric("Chunks", len(doc.chunks))
        full_doc_chunks = [c for c in doc.chunks if c.chunk_type == "full_doc"]
        word_count = len(full_doc_chunks[0].content.split()) if full_doc_chunks else 0
        col4.metric("Words", f"{word_count:,}")
        st.caption(f"**File:** {doc.filename}  |  **ID:** {doc.document_id}  |  **Parsed:** {doc.parsed_at}")
        if doc.metadata:
            m = doc.metadata
            if m.get("title"):
                st.markdown(f"**Title:** {m['title']}")
            if m.get("authors"):
                authors = m["authors"] if isinstance(m["authors"], list) else [m["authors"]]
                st.markdown(f"**Authors:** {', '.join(str(a) for a in authors)}")
            if m.get("year"):
                st.markdown(f"**Year:** {m['year']}")
            if m.get("abstract"):
                with st.expander("Abstract"):
                    st.write(m["abstract"])

    with tab_sections:
        for s in doc.sections:
            indent = "　" * (s.level - 1)
            label = f"{indent}{'#' * s.level} {s.title or '(untitled)'}"
            with st.expander(label):
                st.markdown(s.content[:600] + ("…" if len(s.content) > 600 else ""))

    with tab_chunks:
        rows = [{"pos": c.position, "type": c.chunk_type,
                 "heading": " > ".join(c.heading_path) if c.heading_path else "—",
                 "tokens": c.token_count,
                 "preview": (c.content[:120] + "…") if len(c.content) > 120 else c.content}
                for c in doc.chunks]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_export:
        json_str = json.dumps(doc_dict, indent=2, ensure_ascii=False)
        st.download_button("⬇ Download JSON", data=json_str,
                           file_name=f"{doc.document_id}_parsed.json",
                           mime="application/json", use_container_width=True)
        with st.expander("Raw JSON", expanded=False):
            st.json(doc_dict)


# ── Structured Results ────────────────────────────────────────────────────────

if st.session_state.get("mode") == "structured" and "sdoc" in st.session_state:
    from docparse.models import StructuredDoc
    from docparse import structurer as _structurer

    sdoc: StructuredDoc = st.session_state["sdoc"]

    tab_overview, tab_sections, tab_chunks, tab_export = st.tabs(
        ["Overview", "Sections", "Chunks", "Export"]
    )

    with tab_overview:
        p = sdoc.profile
        col1, col2, col3 = st.columns(3)
        col1.metric("Doc type", p.doc_type)
        col2.metric("Sections", len(sdoc.sections))
        col3.metric("Chunks", len(sdoc.chunks))

        st.caption(f"**File:** {sdoc.filename}  |  **ID:** {sdoc.document_id}  |  **Parsed:** {sdoc.parsed_at}")

        st.subheader("Document profile")
        lang_badges = " ".join(_lang_badge(l) for l in p.languages)
        st.markdown(f"**Languages:** {lang_badges}", unsafe_allow_html=True)
        st.markdown(f"**Structure:** `{p.structure_pattern}`")
        if p.structure_notes:
            st.info(p.structure_notes)

    with tab_sections:
        # Language filter
        all_langs = ["All"] + _structurer.detected_language_slugs(sdoc)
        lang_filter = st.radio("Filter by language", all_langs, horizontal=True)

        shown = sdoc.sections
        if lang_filter != "All":
            from docparse.structurer import _lang_slug
            shown = [s for s in sdoc.sections
                     if _lang_slug(s.language) == lang_filter or s.language in ("both", "unknown")]

        st.caption(f"{len(shown)} section(s)")
        for s in shown:
            badge = _lang_badge(s.language)
            indent = "　" * (s.level - 1)
            label = f"{indent}{'#' * s.level} {s.label or s.section_id}"
            with st.expander(label):
                st.markdown(badge, unsafe_allow_html=True)
                st.markdown(s.content[:600] + ("…" if len(s.content) > 600 else ""))

    with tab_chunks:
        rows = [{"pos": c.position, "type": c.chunk_type, "lang": c.language,
                 "heading": " > ".join(c.heading_path) if c.heading_path else "—",
                 "tokens": c.token_count,
                 "preview": (c.content[:120] + "…") if len(c.content) > 120 else c.content}
                for c in sdoc.chunks]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with tab_export:
        stem = Path(sdoc.filename).stem

        st.markdown("**Download outputs**")
        col_a, col_b = st.columns(2)

        with col_a:
            st.download_button(
                "⬇ Raw markdown",
                data=sdoc.raw_markdown,
                file_name=f"{stem}_raw.md",
                mime="text/markdown",
                use_container_width=True,
            )
            combined_md = _structurer.to_combined_markdown(sdoc)
            st.download_button(
                "⬇ Structured markdown (combined)",
                data=combined_md,
                file_name=f"{stem}_structured.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with col_b:
            structured_json = _structurer.to_structured_json(sdoc)
            st.download_button(
                "⬇ Structured JSON",
                data=structured_json,
                file_name=f"{stem}_structured.json",
                mime="application/json",
                use_container_width=True,
            )

        lang_slugs = _structurer.detected_language_slugs(sdoc)
        if lang_slugs:
            st.markdown("**Per-language markdown**")
            lang_cols = st.columns(len(lang_slugs))
            for col, lang_slug in zip(lang_cols, lang_slugs):
                lang_md = _structurer.to_language_markdown(sdoc, lang_slug)
                if lang_md:
                    col.download_button(
                        f"⬇ {lang_slug.upper()} only",
                        data=lang_md,
                        file_name=f"{stem}_{lang_slug}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )

        st.divider()
        with st.expander("Raw JSON", expanded=False):
            st.json(json.loads(structured_json))
