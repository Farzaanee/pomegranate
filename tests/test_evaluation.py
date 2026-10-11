import json
from pathlib import Path

import pytest

from investment_rag.evaluation import (
    PASS_THRESHOLD,
    EvalCase,
    load_cases,
    quote_supported,
    run_case,
    run_evaluation,
)
from investment_rag.models import Chunk, SearchResult
from investment_rag.profile import UserProfile

UK_TEXT = ("Diversification spreads your money across different investments, which lowers risk. "
           "Check the fees and charges before you invest, and pay off high-interest debt first.")
CHUNK_UK = Chunk("uk-0", UK_TEXT, "FCA InvestSmart", "UK", "https://fca/div", "Diversification", 0)
CHUNK_EU = Chunk("eu-0", "MiFID II sets investor protection rules.", "ESMA Investor Corner", "EU",
                 "https://esma/mifid", "MiFID II", 0)
PROFILE = UserProfile(2500, 3000, "retirement", 15, "medium", "UK")
REC_CASE = EvalCase("rec", "recommendation", ("diversif",), profile=PROFILE)
CMP_CASE = EvalCase("cmp", "comparison", ("mifid",), topic="investor protections")


class RegionRetriever:
    """Returns the one chunk that belongs to the requested region."""

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        """Return the region's chunk (UK for ``None``)."""
        return [SearchResult(CHUNK_EU if region == "EU" else CHUNK_UK, 0.1)]


class EmptyRetriever:
    """Retrieves nothing."""

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        """Return no results."""
        return []


class FakeLLM:
    """Answers both request paths with canned payloads."""

    def __init__(self, recommendation: dict[str, object] | None = None,
                 comparison: dict[str, object] | None = None) -> None:
        """Store the payloads to return."""
        self.recommendation = recommendation or {}
        self.comparison = comparison or {}

    def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Return the canned recommendation payload."""
        return self.recommendation

    def compare(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Return the canned comparison payload."""
        return self.comparison


def good_recommendation() -> dict[str, object]:
    """A payload that should pass every check."""
    return {
        "summary": "A diversified, low-cost mix generally suits a long horizon.",
        "suitable_options": [{"vehicle_type": "Broad index funds", "why_it_fits": "Spreads risk.",
                              "tradeoffs": "Values can fall as well as rise.", "citation_labels": ["1"]}],
        "reasoning": ["Diversification lowers risk. [1]"],
        "risks": ["Markets can fall."],
        "caveats": ["Check fees and pay off high-interest debt first."],
        "citations": [{"label": "1", "quote": "Diversification spreads your money across different investments"}],
    }


def good_comparison() -> dict[str, object]:
    """A comparison payload citing one passage per region."""
    return {
        "eu_summary": "MiFID II sets baseline protections.",
        "uk_summary": "The FCA stresses diversification.",
        "key_difference": "The EU relies on MiFID II; the UK on FCA rules.",
        "citations": [{"label": "EU-1", "quote": "MiFID II sets investor protection rules."},
                      {"label": "UK-1", "quote": "which lowers risk"}],
    }


def failed(result_checks: list, hard_only: bool = True) -> set[str]:
    """Names of the checks that failed."""
    return {c.name for c in result_checks if not c.passed and (c.hard or not hard_only)}


def test_quote_supported_ignores_case_punctuation_and_spacing() -> None:
    """A verbatim quote matches however the model happened to space or punctuate it."""
    assert quote_supported("DIVERSIFICATION spreads   your money,", UK_TEXT)


def test_quote_supported_accepts_elided_quotes_but_rejects_invented_ones() -> None:
    """Each side of an ellipsis must be real, in the passage."""
    assert quote_supported("Diversification spreads … which lowers risk", UK_TEXT)
    assert not quote_supported("Diversification guarantees … which lowers risk", UK_TEXT)
    assert not quote_supported("", UK_TEXT)


def test_good_recommendation_passes_every_hard_check() -> None:
    """The happy path: grounded, region-pure, disclaimed, and free of hype."""
    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=good_recommendation()))

    assert result.error is None
    assert result.passed, failed(result.checks)
    assert result.stats["labels_valid"] == result.stats["labels_total"] == 1
    assert result.stats["quotes_supported"] == 1


def test_fabricated_citation_label_is_counted_as_a_hallucination() -> None:
    """The raw payload is scored, so an invented label fails even though parsing would drop it."""
    payload = good_recommendation()
    payload["citations"] = [*payload["citations"], {"label": "99", "quote": "made up"}]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert not result.passed
    assert "no_fabricated_citations" in failed(result.checks)
    assert result.stats["labels_total"] == 2 and result.stats["labels_valid"] == 1


def test_quote_not_in_the_cited_passage_fails_even_with_a_valid_label() -> None:
    """A real label with an invented quote is a citation-accuracy failure."""
    payload = good_recommendation()
    payload["citations"] = [{"label": "1", "quote": "Index funds always beat the market."}]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert "quotes_supported" in failed(result.checks)


def test_all_invalid_citations_fail_the_case_without_aborting_the_run() -> None:
    """parse_recommendation refuses an ungrounded answer; the harness records that as a failed check."""
    payload = good_recommendation()
    payload["citations"] = [{"label": "99", "quote": "x"}]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert result.error is None
    assert {"has_citations", "parses_with_valid_citations"} <= failed(result.checks)


def test_option_without_a_valid_citation_label_fails() -> None:
    """Every suitable option must be traceable to at least one real passage."""
    payload = good_recommendation()
    payload["suitable_options"][0]["citation_labels"] = ["99"]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert "options_cited" in failed(result.checks)


@pytest.mark.parametrize("phrase, check", [
    ("Vanguard funds are a good fit.", "no_product_names"),
    ("This is a guaranteed way to grow wealth.", "no_hype"),
    ("It will grow by 7% a year.", "no_return_projection"),
    ("You should invest £500 now.", "no_amount_to_invest"),
])
def test_prohibited_language_fails_the_matching_check(phrase: str, check: str) -> None:
    """Product names, hype, return forecasts, and amounts to invest are each caught."""
    payload = good_recommendation()
    payload["summary"] = phrase

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert check in failed(result.checks)


def test_negated_hype_words_are_not_flagged() -> None:
    """"No investment is guaranteed" is the honest phrasing, so it must not trip the hype check."""
    payload = good_recommendation()
    payload["risks"] = ["Returns are not guaranteed and no investment is risk-free."]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert "no_hype" not in failed(result.checks)


def test_fee_and_debt_awareness_are_reported_but_do_not_gate_the_case() -> None:
    """Soft checks: a recommendation that never mentions fees or debt still passes but is measured."""
    payload = good_recommendation()
    payload["caveats"] = ["Speak to an adviser."]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert result.passed
    assert failed(result.checks, hard_only=False) >= {"mentions_fees", "mentions_debt"}


class LeakyRetriever:
    """Ignores the region filter entirely."""

    def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
        """Return both regions' chunks regardless of ``region``."""
        return [SearchResult(CHUNK_UK, 0.1), SearchResult(CHUNK_EU, 0.2)]


def test_leaky_retriever_is_neutralised_by_routing_and_the_eval_confirms_it() -> None:
    """gather_evidence drops the EU chunk, so the region-purity check still passes end to end."""
    result = run_case(REC_CASE, LeakyRetriever(), FakeLLM(recommendation=good_recommendation()))

    assert "region_pure_evidence" not in failed(result.checks)
    assert result.stats["citations_region_ok"] == result.stats["citations_total"]


def test_region_match_rate_is_reported_for_recommendations() -> None:
    """Every UK-profile citation resolving to a UK passage gives a region match rate of 100%."""
    report = run_evaluation([REC_CASE], RegionRetriever(), FakeLLM(recommendation=good_recommendation()), "fake")

    assert report.summary()["region_match_rate"] == 1.0


def test_comparison_passes_when_both_regions_are_cited_with_real_quotes() -> None:
    """The comparison happy path."""
    result = run_case(CMP_CASE, RegionRetriever(), FakeLLM(comparison=good_comparison()))

    assert result.passed, failed(result.checks)


def test_comparison_citing_only_one_region_fails() -> None:
    """A "difference" grounded in a single region is not a comparison."""
    payload = good_comparison()
    payload["citations"] = [payload["citations"][0]]

    result = run_case(CMP_CASE, RegionRetriever(), FakeLLM(comparison=payload))

    assert "cites_both_regions" in failed(result.checks)


def test_comparison_fails_when_a_region_has_no_evidence() -> None:
    """If retrieval only finds one region, the case fails before the model is asked."""
    class UkOnly:
        """Finds nothing for the EU."""

        def search(self, question: str, limit: int = 4, region: str | None = None) -> list[SearchResult]:
            """Return a UK chunk only for the UK."""
            return [SearchResult(CHUNK_UK, 0.1)] if region == "UK" else []

    result = run_case(CMP_CASE, UkOnly(), FakeLLM(comparison=good_comparison()))

    assert "both_regions_retrieved" in failed(result.checks)


def test_retrieval_only_mode_never_calls_a_model() -> None:
    """With no LLM the harness checks evidence only, and labels the report accordingly."""
    report = run_evaluation([REC_CASE, CMP_CASE], RegionRetriever(), None)

    assert report.mode == "retrieval-only"
    assert report.summary()["quote_support_rate"] is None
    assert "Retrieval-only" in report.to_markdown()


def test_empty_retrieval_fails_the_case() -> None:
    """No evidence means nothing could be grounded."""
    result = run_case(REC_CASE, EmptyRetriever(), FakeLLM(recommendation=good_recommendation()))

    assert not result.passed
    assert "evidence_retrieved" in failed(result.checks)


def test_an_exception_aborts_only_its_own_case() -> None:
    """A model error is recorded on that case and the rest of the run continues."""
    class BrokenLLM(FakeLLM):
        """Raises on the recommendation path only."""

        def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
            """Simulate an API failure."""
            raise RuntimeError("api down")

    report = run_evaluation([REC_CASE, CMP_CASE], RegionRetriever(), BrokenLLM(comparison=good_comparison()), "fake")

    assert report.results[0].error == "RuntimeError: api down"
    assert report.results[1].passed
    assert report.summary()["errors"] == 1


def test_threshold_is_applied_to_the_case_pass_rate() -> None:
    """9 of 10 passing meets the bar; 8 of 10 does not."""
    cases = [REC_CASE] * 10
    good = run_evaluation(cases, RegionRetriever(), FakeLLM(recommendation=good_recommendation()), "fake")
    assert good.meets_threshold() and good.summary()["groundedness_rate"] == 1.0

    mixed = run_evaluation(cases, RegionRetriever(), FakeLLM(recommendation={"summary": "x", "citations": []}), "fake")
    assert not mixed.meets_threshold()
    assert PASS_THRESHOLD == 0.9


def test_report_serialises_to_json_and_markdown() -> None:
    """Both renderers work and include the per-case outcome."""
    report = run_evaluation([REC_CASE], RegionRetriever(), FakeLLM(recommendation=good_recommendation()), "fake")

    assert json.loads(json.dumps(report.to_dict()))["cases"][0]["id"] == "rec"
    assert "| rec | recommendation | PASS |" in report.to_markdown()


def test_committed_eval_set_loads_and_covers_both_regions_and_kinds() -> None:
    """evals/cases.json stays valid, and mixes recommendation and comparison cases for EU and UK."""
    cases = load_cases(Path(__file__).parent.parent / "evals" / "cases.json")

    assert {case.kind for case in cases} == {"recommendation", "comparison"}
    assert {case.profile.region for case in cases if case.profile} == {"EU", "UK"}
    assert len({case.id for case in cases}) == len(cases)


def test_quote_supported_tolerates_scraper_markup_and_a_repaired_chunk_boundary() -> None:
    """Real ESMA text carries CSS-class residue and chunks can start mid-word; faithful quotes must still match."""
    passage = ("learly indicate whether their strategy is active or passive. ESMA does this by: "
               "primary_grey_background Cost and performance white_background work with NCAs")
    assert quote_supported("clearly indicate whether their strategy is active or passive", passage)
    assert quote_supported("ESMA does this by: Cost and performance work with NCAs", passage)
    assert not quote_supported("ESMA guarantees that funds never lose money", passage)


def test_hype_word_describing_what_scammers_claim_is_not_flagged() -> None:
    """Reporting a scam's promise ("guaranteed profits") is the opposite of making it."""
    payload = good_recommendation()
    payload["risks"] = ["Scammers use unrealistic promises like guaranteed trading profits."]

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert "no_hype" not in failed(result.checks)


def test_asserted_hype_is_still_flagged() -> None:
    """The context exemption must not let a plain assertion through."""
    payload = good_recommendation()
    payload["summary"] = "Index funds are a guaranteed way to build wealth."

    result = run_case(REC_CASE, RegionRetriever(), FakeLLM(recommendation=payload))

    assert "no_hype" in failed(result.checks)
