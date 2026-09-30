import pytest

from investment_rag.models import Chunk, SearchResult
from investment_rag.profile import UserProfile
from investment_rag.reasoning import (
    DISCLAIMER,
    EvidencePassage,
    ReasoningAgent,
    ReasoningError,
    build_user_prompt,
    explain_region_difference,
    gather_evidence,
    gather_region_comparison_evidence,
    parse_recommendation,
    parse_region_difference,
)

CHUNK_A = Chunk("uk-0", "Diversification lowers risk.", "MoneyHelper", "UK", "https://mh/div", "Diversification", 0)
CHUNK_B = Chunk("uk-1", "An ISA shelters growth from tax.", "MoneyHelper", "UK", "https://mh/isa", "ISAs", 0)
CHUNK_EU = Chunk("eu-0", "MiFID II sets investor protection rules.", "ESMA Investor Corner", "EU",
                  "https://esma/mifid", "MiFID II", 0)


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
        "suitable_options": [{
            "vehicle_type": "Broad index funds",
            "why_it_fits": "A long horizon can ride out volatility.",
            "tradeoffs": "Values can fall as well as rise.",
            "citation_labels": ["1", "99"],
        }],
        "reasoning": ["Diversification reduces risk. [1]"],
        "risks": ["Markets can fall."],
        "caveats": ["Check any existing high-interest debt first."],
        "citations": [{"label": "1", "quote": "Diversification lowers risk."}],
    })
    agent = ReasoningAgent(FakeRetriever(), llm)

    recommendation = agent.run(profile)

    assert recommendation.summary.startswith("Consider")
    assert recommendation.suitable_options[0].vehicle_type == "Broad index funds"
    assert recommendation.suitable_options[0].citation_labels == ["1"]
    assert recommendation.risks == ["Markets can fall."]
    assert recommendation.caveats[-1] == DISCLAIMER
    assert recommendation.citations[0].source_name == "MoneyHelper"
    assert recommendation.citations[0].url == "https://mh/div"


def test_user_prompt_frames_the_profile_as_synthetic_and_asks_for_suitability() -> None:
    """The prompt must read as an educational suitability question, not an instruction to act."""
    profile = UserProfile(2000, 1000, "retirement", 10, "medium", "UK")

    prompt = build_user_prompt(profile, [EvidencePassage("1", CHUNK_A)])

    assert "Synthetic profile" in prompt
    assert "which types of investment vehicle are generally suitable" in prompt
    assert "suggest an amount to invest" in prompt


def test_reasoning_agent_raises_when_no_evidence_available() -> None:
    agent = ReasoningAgent(EmptyRetriever(), FakeLLM({}))

    with pytest.raises(ReasoningError):
        agent.run(UserProfile(2000, 1000, "retirement", 10, "medium", "UK"))


def test_parse_recommendation_drops_fabricated_citation_label() -> None:
    evidence = [EvidencePassage("1", CHUNK_A)]
    payload = {
        "summary": "x", "suitable_options": [], "reasoning": [], "risks": [], "caveats": [],
        "citations": [{"label": "1", "quote": "ok"}, {"label": "99", "quote": "invented"}],
    }

    recommendation = parse_recommendation(payload, evidence)

    assert [c.label for c in recommendation.citations] == ["1"]


def test_parse_recommendation_always_appends_the_disclaimer_once() -> None:
    """The standing disclaimer is added locally, even if the model already emitted it."""
    evidence = [EvidencePassage("1", CHUNK_A)]
    payload = {
        "summary": "x", "suitable_options": [], "reasoning": [], "risks": [],
        "caveats": [DISCLAIMER], "citations": [{"label": "1", "quote": "ok"}],
    }

    recommendation = parse_recommendation(payload, evidence)

    assert recommendation.caveats == [DISCLAIMER]


def test_parse_recommendation_raises_when_all_citations_invalid() -> None:
    evidence = [EvidencePassage("1", CHUNK_A)]
    payload = {"summary": "x", "suitable_options": [], "reasoning": [], "risks": [], "caveats": [],
               "citations": [{"label": "99", "quote": "invented"}]}

    with pytest.raises(ReasoningError):
        parse_recommendation(payload, evidence)


class LeakyRetriever:
    """Simulates a misconfigured store that ignores the region filter it was asked for."""

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        return [SearchResult(CHUNK_A, 0.1), SearchResult(CHUNK_EU, 0.2)]


def test_gather_evidence_drops_out_of_region_results_even_if_the_retriever_leaks_them() -> None:
    """Phase 3: routing must not trust the retriever's region filter alone."""
    profile = UserProfile(2000, 1000, "retirement", 10, "medium", "UK")

    evidence = gather_evidence(LeakyRetriever(), profile)

    assert all(item.chunk.region == "UK" for item in evidence)
    assert CHUNK_EU.id not in {item.chunk.id for item in evidence}


class TwoRegionRetriever:
    """Returns one region-appropriate chunk per requested region, honoring the filter correctly."""

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        if region == "UK":
            return [SearchResult(CHUNK_A, 0.1)]
        if region == "EU":
            return [SearchResult(CHUNK_EU, 0.1)]
        return [SearchResult(CHUNK_A, 0.1), SearchResult(CHUNK_EU, 0.1)]


def test_same_profile_evaluated_under_both_regions_never_mixes_sources() -> None:
    """Comparison scenario: swapping only the region must never leak the other region's evidence."""
    retriever = TwoRegionRetriever()
    uk_profile = UserProfile(2000, 1000, "retirement", 10, "medium", "UK")
    eu_profile = UserProfile(2000, 1000, "retirement", 10, "medium", "EU")

    uk_evidence = gather_evidence(retriever, uk_profile)
    eu_evidence = gather_evidence(retriever, eu_profile)

    assert {item.chunk.region for item in uk_evidence} == {"UK"}
    assert {item.chunk.region for item in eu_evidence} == {"EU"}


def test_gather_region_comparison_evidence_labels_each_passage_by_region() -> None:
    evidence = gather_region_comparison_evidence(TwoRegionRetriever(), "investor protections")

    labels_by_region = {item.chunk.region: item.label for item in evidence}
    assert labels_by_region["UK"].startswith("UK-")
    assert labels_by_region["EU"].startswith("EU-")


def test_gather_region_comparison_evidence_drops_leaked_out_of_region_results() -> None:
    evidence = gather_region_comparison_evidence(LeakyRetriever(), "investor protections")

    assert all(item.label.startswith(item.chunk.region) for item in evidence)


class FakeCompareLLM:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def compare(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        return self.payload


def test_explain_region_difference_returns_grounded_comparison() -> None:
    llm = FakeCompareLLM({
        "eu_summary": "MiFID II sets baseline investor protection rules across the EU.",
        "uk_summary": "An ISA is a UK-specific tax-advantaged wrapper.",
        "key_difference": "The UK offers a tax-advantaged account wrapper the EU regime doesn't.",
        "citations": [{"label": "EU-1", "quote": "MiFID II sets investor protection rules."},
                       {"label": "UK-1", "quote": "An ISA shelters growth from tax."}],
    })

    difference = explain_region_difference(TwoRegionRetriever(), llm, "ISA vs MiFID II")

    assert difference.key_difference.startswith("The UK offers")
    regions_cited = {citation.region for citation in difference.citations}
    assert regions_cited == {"EU", "UK"}


def test_parse_region_difference_drops_citation_whose_label_region_does_not_match_the_chunk() -> None:
    """The model can't launder a UK claim through an EU-prefixed label or vice versa."""
    evidence = [EvidencePassage("EU-1", CHUNK_EU), EvidencePassage("UK-1", CHUNK_A)]
    payload = {
        "eu_summary": "x", "uk_summary": "y", "key_difference": "z",
        "citations": [{"label": "UK-1", "quote": "mislabeled"}],
    }
    # Swap the label a UK chunk resolves under so it looks EU-prefixed but resolves to a UK chunk.
    mismatched = [EvidencePassage("EU-1", CHUNK_A)]
    payload_mismatched = {
        "eu_summary": "x", "uk_summary": "y", "key_difference": "z",
        "citations": [{"label": "EU-1", "quote": "mislabeled"}],
    }

    result = parse_region_difference(payload, evidence)
    assert [c.label for c in result.citations] == ["UK-1"]

    with pytest.raises(ReasoningError):
        parse_region_difference(payload_mismatched, mismatched)


def test_explain_region_difference_raises_when_no_evidence_available() -> None:
    class EmptyBothRetriever:
        def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
            return []

    with pytest.raises(ReasoningError):
        explain_region_difference(EmptyBothRetriever(), FakeCompareLLM({}), "any topic")
