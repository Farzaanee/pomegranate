import pytest

from investment_rag.chunking import chunk_document
from investment_rag.models import SourceDocument


def test_chunks_keep_metadata_and_overlap() -> None:
    document = SourceDocument(
        "ESMA Investor Corner", "EU", "https://example.test/risk", "Risk basics",
        "Diversification can reduce concentration risk. Asset classes behave differently. "
        "Your time horizon affects how much volatility you can tolerate.",
    )

    chunks = chunk_document(document, max_chars=75, overlap_chars=20)

    assert len(chunks) == 3
    assert all(chunk.source_name == "ESMA Investor Corner" for chunk in chunks)
    assert all(chunk.region == "EU" for chunk in chunks)
    assert all(chunk.url == document.url for chunk in chunks)
    assert chunks[0].id.endswith("-0")
    assert "risk." in chunks[1].text.lower()


def test_chunking_rejects_invalid_overlap() -> None:
    document = SourceDocument("FCA InvestSmart", "UK", "https://example.test", "Title", "One sentence.")

    with pytest.raises(ValueError):
        chunk_document(document, max_chars=100, overlap_chars=100)
