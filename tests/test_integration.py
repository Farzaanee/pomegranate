"""End-to-end checks: real chunking output and real Chroma retrieval.

These use a deterministic fake embedder so nothing is downloaded, but they
exercise the actual ``chunk_document`` logic and a real on-disk Chroma store
through ``Retriever.index`` / ``Retriever.search``.
"""

from investment_rag.chunking import chunk_document
from investment_rag.models import Chunk, SourceDocument
from investment_rag.retrieval import Retriever

SAMPLE_TEXT = (
    "Diversification means spreading your money across different investments. "
    "It lowers the impact of any single one falling in value. "
    "Shares, bonds and cash tend to behave differently over time. "
    "Your time horizon changes how much risk you can afford to take. "
    "A longer horizon lets you ride out short-term volatility."
)


class BagOfWordsEmbedder:
    """Counts keyword hits so related passages land near each other."""

    VOCAB = ("diversification", "risk", "bond", "isa", "tax", "horizon", "fees")

    def encode(self, texts: list[str]) -> list[list[float]]:
        return [[float(text.lower().count(word)) for word in self.VOCAB] for text in texts]


def test_chunk_document_respects_size_and_overlap_and_keeps_metadata() -> None:
    document = SourceDocument(
        "MoneyHelper", "UK", "https://example.test/basics", "Investing basics", SAMPLE_TEXT
    )

    chunks = chunk_document(document, max_chars=140, overlap_chars=40)

    assert len(chunks) > 1
    # every chunk is attributable
    assert all(c.source_name == "MoneyHelper" and c.region == "UK" for c in chunks)
    assert all(c.url == document.url and c.title == document.title for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    # chunk ids are "<doc digest>-<index>"
    assert chunks[0].id.rsplit("-", 1)[0] == chunks[1].id.rsplit("-", 1)[0]
    # chunks stay near the requested size (allow one sentence of spillover)
    assert all(len(c.text) <= 220 for c in chunks)
    # consecutive chunks share the carried-over tail of the previous one
    assert chunks[0].text[-20:].strip() in chunks[1].text


def test_retriever_ranks_relevant_passage_first(tmp_path) -> None:
    chunks = [
        Chunk("uk-0", "Diversification lowers risk by spreading your money.",
              "MoneyHelper", "UK", "https://mh/diversification", "UK basics", 0),
        Chunk("uk-1", "An ISA shelters investment growth from tax.",
              "MoneyHelper", "UK", "https://mh/isa", "ISAs", 1),
    ]
    retriever = Retriever(persist_dir=tmp_path / "chroma", embedder=BagOfWordsEmbedder())
    retriever.index(chunks)

    results = retriever.search("how does diversification reduce risk?", limit=2)

    assert results[0].chunk.id == "uk-0"
    assert results[0].chunk.url == "https://mh/diversification"
    assert results[0].distance <= results[1].distance


def test_retriever_region_filter_excludes_other_regions(tmp_path) -> None:
    chunks = [
        Chunk("uk-0", "Diversification reduces risk for UK investors.",
              "MoneyHelper", "UK", "https://mh/uk", "UK basics", 0),
        Chunk("eu-0", "Diversification reduces concentration risk for EU investors.",
              "ESMA Investor Corner", "EU", "https://esma/eu", "EU basics", 0),
    ]
    retriever = Retriever(persist_dir=tmp_path / "chroma", embedder=BagOfWordsEmbedder())
    retriever.index(chunks)

    results = retriever.search("diversification and risk", limit=5, region="UK")

    assert results, "expected at least one UK result"
    assert {r.chunk.region for r in results} == {"UK"}


def test_retriever_index_is_idempotent(tmp_path) -> None:
    chunk = Chunk("uk-0", "Bonds pay regular income.", "MoneyHelper", "UK",
                  "https://mh/bonds", "Bonds", 0)
    retriever = Retriever(persist_dir=tmp_path / "chroma", embedder=BagOfWordsEmbedder())

    retriever.index([chunk])
    retriever.index([chunk])

    assert retriever.collection.count() == 1
