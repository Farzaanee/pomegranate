from investment_rag.retrieval import Retriever


class FakeEmbedder:
    def encode(self, texts):
        return [[0.1, 0.2] for _ in texts]


class FakeCollection:
    def __init__(self):
        self.query_kwargs = None

    def query(self, **kwargs):
        self.query_kwargs = kwargs
        return {
            "ids": [["risk-0"]],
            "documents": [["Diversification spreads investments across assets."]],
            "metadatas": [[{
                "source_name": "MoneyHelper", "region": "UK", "url": "https://example.test/risk",
                "title": "Risk basics", "chunk_index": 0,
            }]],
            "distances": [[0.12]],
        }


def test_search_returns_citable_passage_and_applies_region_filter() -> None:
    retriever = Retriever.__new__(Retriever)
    retriever.embedder = FakeEmbedder()
    retriever.collection = FakeCollection()

    results = retriever.search("How does diversification reduce risk?", region="UK")

    assert results[0].chunk.source_name == "MoneyHelper"
    assert results[0].chunk.url == "https://example.test/risk"
    assert results[0].chunk.region == "UK"
    assert retriever.collection.query_kwargs["where"] == {"region": "UK"}