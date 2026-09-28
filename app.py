"""Streamlit interface for the grounded retrieval pipeline (Phase 1) and the
citation-backed reasoning agent built on top of it (Phase 2).

Run locally with ``streamlit run app.py``. On Hugging Face Spaces the Streamlit
SDK reads the YAML header in ``README.md`` and runs this file automatically.

The Chroma index is built offline by ``investment-rag build`` and committed under
``data/index/``; this app just opens it. "Retrieval search" exposes semantic
search directly, with the same region filter as ``investment-rag query``.
"Grounded recommendation" additionally sends the retrieved evidence to Claude
(``investment_rag.reasoning``) to produce a cited, plain-language recommendation
for a user profile, provided an ``ANTHROPIC_API_KEY`` is configured. Nothing in
the knowledge base is re-embedded at startup — only the user's query is
embedded, at search time.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from investment_rag.models import SearchResult, SourceDocument
from investment_rag.profile import GOALS, RISK_TOLERANCES, UserProfile
from investment_rag.reasoning import ClaudeRecommendationLLM, Recommendation, ReasoningAgent, ReasoningError
from investment_rag.retrieval import Retriever

RAW_DIR = Path(__file__).parent / "data" / "raw"
INDEX_DIR = Path(__file__).parent / "data" / "index"
DISCLAIMER = (
    "General information, not regulated financial advice. Passages are retrieved "
    "verbatim from official public sources and are not investment recommendations."
)


def load_documents(directory: Path) -> list[SourceDocument]:
    """Load every cleaned source document written by ``investment-rag collect``."""
    return [
        SourceDocument(**json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(directory.glob("*.json"))
    ]


@st.cache_resource(show_spinner="Opening the search index…")
def load_retriever() -> Retriever:
    """Open the committed Chroma index from a writable temp copy.

    The index ships in the repo (``data/index/``), so startup does no chunking or
    document embedding — it only copies ~8 MB of files and opens the store. The
    copy keeps Chroma's journal files out of the read-only deployment checkout.
    Cached for the life of the Streamlit process.
    """
    working_copy = Path(tempfile.mkdtemp(prefix="chroma-")) / "index"
    shutil.copytree(INDEX_DIR, working_copy)
    return Retriever(persist_dir=working_copy)


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


def get_api_key() -> str | None:
    """Resolve the Anthropic API key from Streamlit secrets, falling back to the environment."""
    try:
        if st.secrets.get("ANTHROPIC_API_KEY"):
            return str(st.secrets["ANTHROPIC_API_KEY"])
    except Exception:
        pass
    return os.environ.get("ANTHROPIC_API_KEY")


def has_api_key() -> bool:
    """Report whether an Anthropic API key is available from secrets or the environment."""
    return bool(get_api_key())


@st.cache_resource(show_spinner=False)
def load_reasoning_agent(_retriever: Retriever) -> ReasoningAgent:
    """Build the reasoning agent with the API key resolved from secrets or the environment.

    The leading underscore on ``_retriever`` tells Streamlit not to hash it (a
    live Chroma client isn't hashable) while still caching one agent per process.
    """
    return ReasoningAgent(_retriever, ClaudeRecommendationLLM(api_key=get_api_key()))


def render_recommendation(recommendation: Recommendation) -> None:
    """Render a grounded suitability analysis with its reasoning, risks, and citations."""
    st.subheader("Suitability analysis")
    st.write(recommendation.summary)
    if recommendation.suitable_options:
        st.markdown("**Vehicle types that fit this profile**")
        for option in recommendation.suitable_options:
            labels = " ".join(f"[{label}]" for label in option.citation_labels)
            with st.expander(f"{option.vehicle_type} {labels}".strip()):
                st.markdown(f"**Why it fits:** {option.why_it_fits}")
                st.markdown(f"**Tradeoffs:** {option.tradeoffs}")
    st.markdown("**Reasoning**")
    for step in recommendation.reasoning:
        st.markdown(f"- {step}")
    if recommendation.risks:
        st.markdown("**Risks**")
        for risk in recommendation.risks:
            st.markdown(f"- {risk}")
    if recommendation.caveats:
        st.markdown("**Caveats**")
        for caveat in recommendation.caveats:
            st.markdown(f"- {caveat}")
    st.markdown("**Sources**")
    for citation in recommendation.citations:
        with st.expander(f"[{citation.label}] {citation.source_name} · {citation.region}"):
            st.write(f"“{citation.quote}”")
            st.markdown(f"[{citation.url}]({citation.url})")


def render_retrieval_mode(retriever: Retriever) -> None:
    """Draw the Phase 1 semantic-search UI and run a search on submission."""
    with st.sidebar:
        region_choice = st.radio("Region filter", ["All", "EU", "UK"], horizontal=True)
        limit = st.slider("Passages to return", min_value=1, max_value=8, value=4)

    question = st.text_input(
        "Ask an investing-education question",
        placeholder="Why does diversification matter when investing?",
    )
    if not question:
        return

    region = None if region_choice == "All" else region_choice
    results = retriever.search(question, limit=limit, region=region)
    if not results:
        st.info("No passages matched. Try rephrasing, or widen the region filter.")
        return

    st.subheader(f"{len(results)} retrieved passage(s)")
    for result in results:
        render_result(result)


def render_recommendation_mode(retriever: Retriever) -> None:
    """Draw the Phase 2 profile form and run the reasoning agent on submission."""
    if not has_api_key():
        st.warning(
            "No Anthropic API key configured (set `ANTHROPIC_API_KEY` as a Streamlit secret or "
            "environment variable) — the reasoning layer needs one to call Claude."
        )
        return

    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox("Region", ["UK", "EU"])
            goal = st.selectbox("Goal", GOALS, format_func=lambda value: value.replace("_", " ").title())
            timeline_years = st.number_input("Timeline (years)", min_value=1, max_value=50, value=10)
        with col2:
            risk_tolerance = st.selectbox("Risk tolerance", RISK_TOLERANCES)
            monthly_income = st.number_input("Monthly income", min_value=0.0, value=2000.0, step=100.0)
            investable_amount = st.number_input("Amount available to invest", min_value=0.0, value=1000.0, step=100.0)
        submitted = st.form_submit_button("Get a grounded recommendation")

    if not submitted:
        return

    profile = UserProfile(monthly_income, investable_amount, goal, int(timeline_years), risk_tolerance, region)
    agent = load_reasoning_agent(retriever)
    with st.spinner("Retrieving evidence and reasoning over it…"):
        try:
            recommendation = agent.run(profile)
        except ReasoningError as error:
            st.error(str(error))
            return
    render_recommendation(recommendation)


def main() -> None:
    """Draw the sidebar and run whichever mode (retrieval or recommendation) is selected."""
    st.set_page_config(page_title="Grounded Investment RAG", page_icon="\U0001f4c8")
    st.title("\U0001f4c8 Grounded Investment Research")
    st.caption(DISCLAIMER)

    documents = load_documents(RAW_DIR)
    sources = sorted({(document.source_name, document.region) for document in documents})
    with st.sidebar:
        st.header("Knowledge base")
        for source_name, region in sources:
            st.markdown(f"- **{source_name}** ({region})")
        st.caption(
            f"{len(sources)} official source(s) across {len(documents)} indexed page(s). "
            "Phase 2 adds a reasoning agent on top of Phase 1 retrieval — see project-plan.md."
        )
        mode = st.radio("Mode", ["Retrieval search", "Grounded recommendation"])

    retriever = load_retriever()
    if mode == "Retrieval search":
        render_retrieval_mode(retriever)
    else:
        render_recommendation_mode(retriever)


if __name__ == "__main__":
    main()
