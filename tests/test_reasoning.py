import pytest

from investment_rag.models import Chunk, SearchResult
from investment_rag.profile import UserProfile
from investment_rag.reasoning import EvidencePassage, ReasoningAgent, ReasoningError, gather_evidence, parse_recommendation

CHUNK_A = Chunk("uk-0", "Diversification lowers risk.", "MoneyHelper", "UK", "https://mh/div", "Diversification", 0)
CHUNK_B = Chunk("uk-1", "An ISA shelters growth from tax.", "MoneyHelper", "UK", "https://mh/isa", "ISAs", 0)


class FakeRetriever:
    """Returns CHUNK_A for every query; queries mentioning ISA also return CHUNK_B."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, str | None]] = []

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        self.calls.append((question, limit, region))
        results = [SearchResult(CHUNK_A, 0.1)]
        if "ISA" in question:
            results.append(SearchResult(CHUNK_B, 0.2))
        return results


class EmptyRetriever:
    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        return []


class FakeLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls: list[tuple[str, str]] = []

    def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        self.calls.append((system_prompt, user_prompt))
        return self.payload


def test_gather_evidence_dedupes_and_labels_sequentially() -> None:
    profile = UserProfile(2000, 1000, "retirement", 10, "medium", "UK")
    retriever = FakeRetriever()

    evidence = gather_evidence(retriever, profile)

    assert [item.chunk.id for item in evidence] == ["uk-0", "uk-1"]
    assert [item.label for item in evidence] == ["1", "2"]
    assert all(region == "UK" for _, _, region in retriever.calls)


def test_reasoning_agent_returns_grounded_recommendation() -> None:
    profile = UserProfile(2000, 1000, "retirement", 10, "medium", "UK")
    llm = FakeLLM({
        "summary": "Consider a diversified, low-cost approach.",
        "reasoning_steps": ["Diversification reduces risk. [1]"],
        "considerations": ["Check any existing high-interest debt first."],
        "citations": [{"label": "1", "quote": "Diversification lowers risk."}],
    })
    agent = ReasoningAgent(FakeRetriever(), llm)

    recommendation = agent.run(profile)

    assert recommendation.summary.startswith("Consider")
    assert recommendation.citations[0].source_name == "MoneyHelper"
    assert recommendation.citations[0].url == "https://mh/div"


def test_reasoning_agent_raises_when_no_evidence_available() -> None:
    agent = ReasoningAgent(EmptyRetriever(), FakeLLM({}))

    with pytest.raises(ReasoningError):
        agent.run(UserProfile(2000, 1000, "retirement", 10, "medium", "UK"))


def test_parse_recommendation_drops_fabricated_citation_label() -> None:
    evidence = [EvidencePassage("1", CHUNK_A)]
    payload = {
        "summary": "x", "reasoning_steps": [], "considerations": [],
        "citations": [{"label": "1", "quote": "ok"}, {"label": "99", "quote": "invented"}],
    }

    recommendation = parse_recommendation(payload, evidence)

    assert [c.label for c in recommendation.citations] == ["1"]


def test_parse_recommendation_raises_when_all_citations_invalid() -> None:
    evidence = [EvidencePassage("1", CHUNK_A)]
    payload = {"summary": "x", "reasoning_steps": [], "considerations": [],
               "citations": [{"label": "99", "quote": "invented"}]}

    with pytest.raises(ReasoningError):
        parse_recommendation(payload, evidence)
