"""Streamlit UI for docparse."""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

from docparse import parser as _parser
from docparse.models import ParsedDoc

st.set_page_config(
    page_title="docparse",
    page_icon="📄",
    layout="wide",
)


def _to_dict(doc: ParsedDoc) -> dict:
    return dataclasses.asdict(doc)


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 docparse")
    st.caption("PDF · DOCX · MD → structured JSON")

    api_key = st.text_input(
        "Mistral API key",
        type="password",
        placeholder="your-api-key-here",
        help="Required for metadata extraction and heading detection.",
    )

    st.divider()

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
        help="Mistral model for heading detection and metadata extraction",
    )

    parse_btn = st.button("Parse", type="primary", use_container_width=True)


# ── Parse ─────────────────────────────────────────────────────────────────────

if parse_btn:
    if not api_key:
        st.error("Enter your Mistral API key in the sidebar.")
    elif not uploaded_file and not local_path.strip():
        st.error("Provide a file upload or local path.")
    else:
        with st.spinner("Parsing document…"):
            try:
                if uploaded_file:
                    suffix = Path(uploaded_file.name).suffix
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    doc = _parser.parse(tmp_path, model=model, api_key=api_key)
                    doc = dataclasses.replace(doc, filename=uploaded_file.name)
                else:
                    doc = _parser.parse(local_path.strip(), model=model, api_key=api_key)

                st.session_state["doc"] = doc
                st.session_state["doc_dict"] = _to_dict(doc)
            except Exception as e:
                st.error(f"Parse failed: {e}")


# ── Results ───────────────────────────────────────────────────────────────────

if "doc" in st.session_state:
    doc: ParsedDoc = st.session_state["doc"]
    doc_dict: dict = st.session_state["doc_dict"]

    tab_overview, tab_sections, tab_chunks, tab_export = st.tabs(
        ["Overview", "Sections", "Chunks", "Export"]
    )

    # ── Overview ──────────────────────────────────────────────────────────────
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
            st.subheader("Metadata")
            m = doc.metadata
            if m.get("title"):
                st.markdown(f"**Title:** {m['title']}")
            if m.get("authors"):
                authors = m["authors"] if isinstance(m["authors"], list) else [m["authors"]]
                st.markdown(f"**Authors:** {', '.join(str(a) for a in authors)}")
            if m.get("year"):
                st.markdown(f"**Year:** {m['year']}")
            if m.get("source"):
                st.markdown(f"**Source:** {m['source']}")
            if m.get("abstract"):
                with st.expander("Abstract"):
                    st.write(m["abstract"])

    # ── Sections ──────────────────────────────────────────────────────────────
    with tab_sections:
        if not doc.sections:
            st.info("No sections detected.")
        else:
            st.caption(f"{len(doc.sections)} sections detected")
            for s in doc.sections:
                indent = "　" * (s.level - 1)
                hashes = "#" * s.level
                label = f"{indent}{hashes} {s.title or '(untitled)'}"
                if s.heading_path:
                    label += f"  ·  *{' > '.join(s.heading_path)}*"

                preview = s.content[:600] + ("…" if len(s.content) > 600 else "")
                with st.expander(label):
                    st.markdown(preview)

    # ── Chunks ────────────────────────────────────────────────────────────────
    with tab_chunks:
        rows = [
            {
                "pos": c.position,
                "type": c.chunk_type,
                "heading": " > ".join(c.heading_path) if c.heading_path else "—",
                "tokens": c.token_count,
                "preview": (c.content[:120] + "…") if len(c.content) > 120 else c.content,
            }
            for c in doc.chunks
        ]
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ── Export ────────────────────────────────────────────────────────────────
    with tab_export:
        json_str = json.dumps(doc_dict, indent=2, ensure_ascii=False)

        st.download_button(
            label="⬇ Download JSON",
            data=json_str,
            file_name=f"{doc.document_id}_parsed.json",
            mime="application/json",
            use_container_width=True,
        )

        st.divider()
        with st.expander("Raw JSON", expanded=False):
            st.json(doc_dict)
