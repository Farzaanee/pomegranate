"""Streamlit interface for the Phase 1 grounded retrieval pipeline.

Run locally with ``streamlit run app.py``. On Hugging Face Spaces the Streamlit
SDK reads the YAML header in ``README.md`` and runs this file automatically.

The app indexes the source documents committed under ``data/raw/`` into a fresh
Chroma store on startup, then exposes semantic search with the same region
filter as the ``investment-rag query`` command.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import streamlit as st

from investment_rag.chunking import chunk_documents
from investment_rag.models import SearchResult, SourceDocument
from investment_rag.retrieval import Retriever

RAW_DIR = Path(__file__).parent / "data" / "raw"
DISCLAIMER = (
    "Educational only — not regulated financial advice. Passages are retrieved "
    "verbatim from official public sources and are not investment recommendations."
)


def load_documents(directory: Path) -> list[SourceDocument]:
    """Load every cleaned source document written by ``investment-rag collect``."""
    return [
        SourceDocument(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]


@st.cache_resource(show_spinner="Loading the embedding model and building the index…")
def build_retriever() -> Retriever:
    """Chunk the committed documents and index them in a throwaway Chroma store.

    Cached for the life of the Streamlit process, so the embedding model loads
    and the index builds once rather than on every widget interaction.
    """
    retriever = Retriever(persist_dir=tempfile.mkdtemp(prefix="chroma-"))
    retriever.index(chunk_documents(load_documents(RAW_DIR)))
    return retriever


def render_result(result: SearchResult) -> None:
    """Render one retrieved passage with its provenance and similarity score."""
    chunk = result.chunk
    st.markdown(
        f"**{chunk.source_name}** · {chunk.region} · "
        f"cosine distance `{result.distance:.3f}`"
    )
    st.markdown(f"*{chunk.title}*")
    st.write(chunk.text)
    st.markdown(f"[{chunk.url}]({chunk.url})")
    st.divider()


def main() -> None:
    """Draw the retrieval UI and run a search when the user submits a question."""
    st.set_page_config(page_title="Grounded Investment RAG", page_icon="\U0001f4c8")
    st.title("\U0001f4c8 Grounded Investment Research — retrieval demo")
    st.caption(DISCLAIMER)

    documents = load_documents(RAW_DIR)
    with st.sidebar:
        st.header("Knowledge base")
        for document in documents:
            st.markdown(f"- **{document.source_name}** ({document.region})")
        st.caption(
            f"{len(documents)} official source document(s) indexed. "
            "Phase 1 of the project plan — retrieval only, no reasoning layer yet."
        )
        region_choice = st.radio("Region filter", ["All", "EU", "UK"], horizontal=True)
        limit = st.slider("Passages to return", min_value=1, max_value=8, value=4)

    question = st.text_input(
        "Ask an investing-education question",
        placeholder="Why does diversification matter when investing?",
    )
    if not question:
        return

    retriever = build_retriever()
    region = None if region_choice == "All" else region_choice
    results = retriever.search(question, limit=limit, region=region)
    if not results:
        st.info("No passages matched. Try rephrasing, or widen the region filter.")
        return

    st.subheader(f"{len(results)} retrieved passage(s)")
    for result in results:
        render_result(result)


if __name__ == "__main__":
    main()
