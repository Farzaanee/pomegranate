"""Evaluation harness (Phase 4): measures groundedness, citation accuracy, and
hallucination rate of the reasoning layer against the documents it retrieved.

Each case in an eval set is run through the *same* evidence-gathering, prompt, and
parsing functions the product uses, but the harness keeps the raw model payload
that ``parse_recommendation`` / ``parse_region_difference`` would otherwise
silently clean up. That lets it report how often the model *tried* to cite
something that was not retrieved (the hallucination rate), rather than only
what survived validation.

Checks are deterministic string/structure checks, not an LLM judge, so a run is
cheap to repeat and every failure can be traced to a specific passage. They are
proxies: a quote that is verbatim in the cited passage is evidence of
groundedness, not proof the claim around it is faithful. ``docs/evaluation.md``
spells out what this does and does not establish.
"""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .profile import REGIONS, UserProfile
from .reasoning import (
    DISCLAIMER,
    MAX_CITATIONS,
    MAX_DIFFERENCE_CITATIONS,
    REGION_DIFFERENCE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    EvidencePassage,
    ReasoningError,
    RecommendationLLM,
    RegionDifferenceLLM,
    build_region_difference_prompt,
    build_user_prompt,
    gather_evidence,
    gather_region_comparison_evidence,
    parse_recommendation,
    parse_region_difference,
)
from .retrieval import Retriever

# The project-plan acceptance bar: "correctly citing sources in the large majority of test cases".
PASS_THRESHOLD = 0.9

KIND_RECOMMENDATION = "recommendation"
KIND_COMPARISON = "comparison"

# Provider names and tickers the system prompt forbids; a hit means the model named a product.
PRODUCT_PATTERN = re.compile(
    r"\b(vanguard|ishares|blackrock|fidelity|hargreaves|trading ?212|etoro|revolut|freetrade|nutmeg|"
    r"amundi|lyxor|xtrackers|scalable capital|trade republic|vwrl|vusa|cspx|swda)\b",
    re.IGNORECASE,
)
# Promotional language. Matches preceded by a negation ("no guarantee", "not risk-free") are ignored.
HYPE_PATTERN = re.compile(
    r"\b(guarantee[sd]?|risk[- ]free|can'?t lose|cannot lose|get rich|sure thing|surefire|to the moon|"
    r"once[- ]in[- ]a[- ]lifetime)\b",
    re.IGNORECASE,
)
NEGATION_PATTERN = re.compile(r"\b(no|not|never|without|cannot|isn'?t|aren'?t|can'?t|don'?t|doesn'?t)\b", re.IGNORECASE)
# Specific return forecasts ("will grow 7%", "expected return of 6") — the prompt forbids projecting returns.
RETURN_PROJECTION_PATTERN = re.compile(
    r"\b(will|should|expect(?:ed)? to|likely to)\s+(earn|return|grow|make|yield)\b[^.]{0,40}\d+(\.\d+)?\s?%"
    r"|\b(expected|average|annual(?:ised)?) returns? of\s+\d",
    re.IGNORECASE,
)
# A specific amount or allocation to act on ("invest £500", "put 60% in") — the prompt forbids both.
ACTION_AMOUNT_PATTERN = re.compile(
    r"\b(invest|put|allocate|put aside|move)\s+(about\s+|around\s+)?[£€$]\s?\d"
    r"|\b(allocate|put|invest|split)\b[^.]{0,30}\d+\s?%",
    re.IGNORECASE,
)
FEE_PATTERN = re.compile(r"\b(fees?|costs?|charges?)\b", re.IGNORECASE)
DEBT_PATTERN = re.compile(r"\b(debts?|borrow\w*|credit cards?|loans?|overdrafts?)\b", re.IGNORECASE)
# A quote fragment counts as supported when this share of its words appear, in order, in the passage.
QUOTE_MATCH_THRESHOLD = 0.9
# Scraper residue in cleaned ESMA pages (e.g. "primary_grey_background"); a faithful quote omits it.
MARKUP_TOKEN_PATTERN = re.compile(r"\b\w+_(?:background|colou?r)\b", re.IGNORECASE)
# A hype word near these is the model describing what scammers claim, not asserting it.
WARNING_CONTEXT_PATTERN = re.compile(
    r"\b(scams?|scammers?|fraud\w*|promis\w*|claims?|claimed|claiming|unrealistic|exaggerat\w*|misleading|"
    r"warns?|warning|too good|pressure)\b",
    re.IGNORECASE,
)
ELLIPSIS_PATTERN = re.compile(r"\[\s*(?:\.{3}|…)\s*\]|\.{3}|…")


class EvalLLM(RecommendationLLM, RegionDifferenceLLM, Protocol):
    """An LLM that can answer both request paths (``ClaudeRecommendationLLM`` does)."""


@dataclass(frozen=True)
class EvalCase:
    """One evaluation scenario: a profile to advise, or a topic to compare across regions.

    ``expected_terms`` are lowercase substrings, any one of which should appear in
    the retrieved evidence — a cheap check that retrieval surfaced on-topic material.
    """

    id: str
    kind: str
    expected_terms: tuple[str, ...]
    profile: UserProfile | None = None
    topic: str | None = None


@dataclass(frozen=True)
class Check:
    """One named pass/fail check. ``hard`` checks gate the case; soft ones are only reported."""

    name: str
    passed: bool
    hard: bool = True
    detail: str = ""


@dataclass
class CaseResult:
    """Outcome of one case: its checks, citation counters, and the error if the run aborted."""

    case_id: str
    kind: str
    checks: list[Check] = field(default_factory=list)
    stats: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def passed(self) -> bool:
        """A case passes when it did not abort and every hard check passed."""
        return self.error is None and all(check.passed for check in self.checks if check.hard)

    def add(self, name: str, passed: bool, hard: bool = True, detail: str = "") -> None:
        """Record one check result."""
        self.checks.append(Check(name, passed, hard, detail))

    def bump(self, **counts: int) -> None:
        """Add to the named citation counters."""
        for key, value in counts.items():
            self.stats[key] = self.stats.get(key, 0) + value


@dataclass(frozen=True)
class EvalReport:
    """All case results from one run, with aggregate metrics and renderers."""

    results: list[CaseResult]
    mode: str
    model: str

    def _rate(self, numerator: str, denominator: str) -> float | None:
        """Sum two counters across cases and divide; ``None`` when nothing was counted."""
        top = sum(result.stats.get(numerator, 0) for result in self.results)
        bottom = sum(result.stats.get(denominator, 0) for result in self.results)
        return top / bottom if bottom else None

    def _check_rate(self, name: str) -> float | None:
        """Share of cases that ran the named check and passed it; ``None`` if none ran it."""
        outcomes = [check.passed for result in self.results for check in result.checks if check.name == name]
        return sum(outcomes) / len(outcomes) if outcomes else None

    def summary(self) -> dict[str, float | int | None]:
        """Aggregate metrics: groundedness (case pass rate) plus the per-citation rates behind it."""
        passed = sum(result.passed for result in self.results)
        return {
            "cases": len(self.results),
            "cases_passed": passed,
            "groundedness_rate": passed / len(self.results) if self.results else None,
            "citation_label_validity": self._rate("labels_valid", "labels_total"),
            "hallucinated_citation_rate": (
                None if self._rate("labels_valid", "labels_total") is None
                else 1 - self._rate("labels_valid", "labels_total")
            ),
            "quote_support_rate": self._rate("quotes_supported", "quotes_checked"),
            "region_match_rate": self._rate("citations_region_ok", "citations_total"),
            "retrieval_hit_rate": self._check_rate("retrieval_on_topic"),
            "fee_awareness_rate": self._check_rate("mentions_fees"),
            "debt_awareness_rate": self._check_rate("mentions_debt"),
            "errors": sum(result.error is not None for result in self.results),
        }

    def meets_threshold(self) -> bool:
        """Whether the case pass rate reaches ``PASS_THRESHOLD``."""
        rate = self.summary()["groundedness_rate"]
        return rate is not None and rate >= PASS_THRESHOLD

    def to_dict(self) -> dict[str, object]:
        """A JSON-serializable form of the whole report."""
        return {
            "mode": self.mode,
            "model": self.model,
            "threshold": PASS_THRESHOLD,
            "meets_threshold": self.meets_threshold(),
            "summary": self.summary(),
            "cases": [
                {
                    "id": result.case_id,
                    "kind": result.kind,
                    "passed": result.passed,
                    "error": result.error,
                    "stats": result.stats,
                    "checks": [
                        {"name": c.name, "passed": c.passed, "hard": c.hard, "detail": c.detail}
                        for c in result.checks
                    ],
                }
                for result in self.results
            ],
        }

    def to_markdown(self) -> str:
        """A human-readable summary table plus the details of every failed check."""

        def percent(value: float | int | None) -> str:
            """Format a rate as a percentage, or ``n/a`` when it was not measured."""
            return "n/a" if value is None else f"{value:.1%}"

        summary = self.summary()
        lines = [
            f"Mode: {self.mode} · model: {self.model or 'none'}",
            *(["", "_Retrieval-only: no LLM was called, so groundedness here means evidence checks only; "
               "citation and language metrics need a full run._"] if self.mode == "retrieval-only" else []),
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Cases passed | {summary['cases_passed']} / {summary['cases']} |",
            f"| Groundedness rate (threshold {PASS_THRESHOLD:.0%}) | {percent(summary['groundedness_rate'])} |",
            f"| Citation label validity | {percent(summary['citation_label_validity'])} |",
            f"| Hallucinated citation rate | {percent(summary['hallucinated_citation_rate'])} |",
            f"| Quote support rate | {percent(summary['quote_support_rate'])} |",
            f"| Region match rate | {percent(summary['region_match_rate'])} |",
            f"| Retrieval hit rate | {percent(summary['retrieval_hit_rate'])} |",
            f"| Fee awareness (soft) | {percent(summary['fee_awareness_rate'])} |",
            f"| Debt awareness (soft) | {percent(summary['debt_awareness_rate'])} |",
            f"| Aborted cases | {summary['errors']} |",
            "",
            "| Case | Kind | Result | Failed checks |",
            "|---|---|---|---|",
        ]
        for result in self.results:
            failed = [f"{c.name}{' (soft)' if not c.hard else ''}: {c.detail}" if c.detail else c.name
                      for c in result.checks if not c.passed]
            if result.error:
                failed.insert(0, f"aborted: {result.error}")
            lines.append(f"| {result.case_id} | {result.kind} | {'PASS' if result.passed else 'FAIL'} | "
                         f"{'; '.join(failed) or '—'} |")
        return "\n".join(lines)


def load_cases(path: str | Path) -> list[EvalCase]:
    """Load eval cases from a JSON file (a list of case objects, see ``evals/cases.json``)."""
    cases = []
    for raw in json.loads(Path(path).read_text(encoding="utf-8")):
        profile = UserProfile(**raw["profile"]) if "profile" in raw else None
        cases.append(EvalCase(raw["id"], raw["kind"], tuple(raw.get("expected_terms", ())), profile,
                              raw.get("topic")))
    return cases


def _squash(text: str) -> str:
    """Lowercase ``text``, drop scraper markup tokens, and collapse non-alphanumerics to single spaces."""
    return re.sub(r"[^a-z0-9]+", " ", MARKUP_TOKEN_PATTERN.sub(" ", text).lower()).strip()


def _repair_cut_edges(passage_words: list[str], fragment: list[str]) -> list[str]:
    """Complete a passage's first/last word when the chunk was cut mid-word and the quote has the whole word.

    Chunks begin and end on character offsets, so a passage can start "learly"
    where the source says "clearly"; a model quoting it will naturally write the
    whole word, which is faithful, not invented.
    """
    words = list(passage_words)
    if not words:
        return words
    for word in fragment:
        if len(words[0]) >= 3 and word != words[0] and word.endswith(words[0]):
            words[0] = word
            break
    for word in fragment:
        if len(words[-1]) >= 3 and word != words[-1] and word.startswith(words[-1]):
            words[-1] = word
            break
    return words


def quote_supported(quote: str, passage_text: str) -> bool:
    """Whether ``quote`` is a faithful excerpt of ``passage_text``.

    Comparison ignores case, punctuation, spacing, and scraper markup tokens. A
    quote containing an ellipsis is split there and every fragment must match. A
    fragment matches when at least ``QUOTE_MATCH_THRESHOLD`` of its words appear,
    in order, in the passage — tolerant of a word completed across a chunk
    boundary or a dropped markup token, but not of an invented sentence.
    """
    passage_words = _squash(passage_text).split()
    fragments = [_squash(part).split() for part in ELLIPSIS_PATTERN.split(quote)]
    fragments = [fragment for fragment in fragments if fragment]
    if not fragments:
        return False
    for fragment in fragments:
        words = _repair_cut_edges(passage_words, fragment)
        matched = sum(block.size for block in SequenceMatcher(None, fragment, words, autojunk=False)
                      .get_matching_blocks())
        if matched / len(fragment) < QUOTE_MATCH_THRESHOLD:
            return False
    return True


def _unnegated(pattern: re.Pattern[str], text: str) -> list[str]:
    """Return matches of ``pattern`` that assert something, i.e. are not negated or reported speech.

    A match is skipped when a negation appears in the 25 characters before it ("no
    guarantee") or a scam/claim/warning word appears within 60 characters either
    side ("promises like guaranteed profits"), since the model is then describing
    what a bad actor says rather than saying it. This is a heuristic.
    """
    found = []
    for match in pattern.finditer(text):
        before = text[max(0, match.start() - 25):match.start()]
        around = text[max(0, match.start() - 60):match.end() + 60]
        if not NEGATION_PATTERN.search(before) and not WARNING_CONTEXT_PATTERN.search(around):
            found.append(match.group(0))
    return found


def _check_language(result: CaseResult, text: str) -> None:
    """Add the hard "avoid hype / products / forecasts / amounts" checks for the model's prose."""
    for name, found in (
        ("no_product_names", PRODUCT_PATTERN.findall(text)),
        ("no_hype", _unnegated(HYPE_PATTERN, text)),
        ("no_return_projection", [m.group(0) for m in RETURN_PROJECTION_PATTERN.finditer(text)]),
        ("no_amount_to_invest", [m.group(0) for m in ACTION_AMOUNT_PATTERN.finditer(text)]),
    ):
        result.add(name, not found, detail=", ".join(str(item) for item in found[:3]))


def _check_retrieval(result: CaseResult, case: EvalCase, evidence: list[EvidencePassage]) -> bool:
    """Add the retrieval checks (non-empty, on-topic) and return whether any evidence exists."""
    result.add("evidence_retrieved", bool(evidence))
    if evidence and case.expected_terms:
        combined = " ".join(item.chunk.text.lower() for item in evidence)
        hit = any(term in combined for term in case.expected_terms)
        result.add("retrieval_on_topic", hit, hard=False,
                   detail="" if hit else f"none of {list(case.expected_terms)} in retrieved text")
    return bool(evidence)


def _score_citations(
    result: CaseResult,
    raw_citations: list[dict[str, str]],
    by_label: dict[str, EvidencePassage],
    required_region: str | None = None,
) -> None:
    """Score raw model citations against the evidence: label validity, quote support, region match.

    ``required_region`` is the one jurisdiction every citation must come from (a
    recommendation's profile region); comparisons cite both regions, so they pass
    ``None`` and are not region-scored per citation.
    """
    labels_total = quotes_checked = quotes_supported = citations_total = region_ok = 0
    fabricated: list[str] = []
    unsupported: list[str] = []
    for raw in raw_citations:
        label = raw.get("label", "")
        labels_total += 1
        passage = by_label.get(label)
        if passage is None:
            fabricated.append(str(label))
            continue
        quotes_checked += 1
        if quote_supported(raw.get("quote", ""), passage.chunk.text):
            quotes_supported += 1
        else:
            unsupported.append(str(label))
        if required_region is not None:
            citations_total += 1
            region_ok += passage.chunk.region == required_region
    result.bump(labels_total=labels_total, labels_valid=labels_total - len(fabricated),
                quotes_checked=quotes_checked, quotes_supported=quotes_supported,
                citations_total=citations_total, citations_region_ok=region_ok)
    result.add("has_citations", labels_total > len(fabricated))
    result.add("no_fabricated_citations", not fabricated, detail=f"labels not in evidence: {fabricated}")
    result.add("quotes_supported", not unsupported, detail=f"quote not found in passage: {unsupported}")


def _run_recommendation(case: EvalCase, retriever: Retriever, llm: EvalLLM | None, result: CaseResult) -> None:
    """Evaluate one profile: retrieval always, then (with an LLM) the full grounded recommendation."""
    assert case.profile is not None
    profile = case.profile
    evidence = gather_evidence(retriever, profile)
    result.add("region_pure_evidence", all(item.chunk.region == profile.region for item in evidence))
    if not _check_retrieval(result, case, evidence) or llm is None:
        return

    payload = llm.recommend(SYSTEM_PROMPT, build_user_prompt(profile, evidence))
    by_label = {item.label: item for item in evidence}
    _score_citations(result, list(payload.get("citations", []))[:MAX_CITATIONS], by_label, profile.region)
    try:
        recommendation = parse_recommendation(payload, evidence)
    except ReasoningError as error:
        result.add("parses_with_valid_citations", False, detail=str(error))
        return
    result.add("parses_with_valid_citations", True)
    result.add("citations_in_profile_region", all(c.region == profile.region for c in recommendation.citations))
    result.add("options_cited", all(option.citation_labels for option in recommendation.suitable_options),
               detail="a suitable option has no valid citation label")
    result.add("disclaimer_present", recommendation.caveats[-1] == DISCLAIMER)

    prose = " ".join([
        recommendation.summary, *recommendation.reasoning, *recommendation.risks,
        *[caveat for caveat in recommendation.caveats if caveat != DISCLAIMER],
        *[f"{o.vehicle_type} {o.why_it_fits} {o.tradeoffs}" for o in recommendation.suitable_options],
    ])
    _check_language(result, prose)
    advice = " ".join([*recommendation.risks, *recommendation.caveats[:-1], *recommendation.reasoning,
                       *[o.tradeoffs for o in recommendation.suitable_options]])
    result.add("mentions_fees", bool(FEE_PATTERN.search(advice)), hard=False)
    result.add("mentions_debt", bool(DEBT_PATTERN.search(advice)), hard=False)


def _run_comparison(case: EvalCase, retriever: Retriever, llm: EvalLLM | None, result: CaseResult) -> None:
    """Evaluate one EU/UK comparison topic: retrieval always, then (with an LLM) the grounded comparison."""
    assert case.topic is not None
    evidence = gather_region_comparison_evidence(retriever, case.topic)
    result.add("region_pure_evidence", all(item.label.startswith(f"{item.chunk.region}-") for item in evidence))
    result.add("both_regions_retrieved", {item.chunk.region for item in evidence} == set(REGIONS))
    if not _check_retrieval(result, case, evidence) or llm is None:
        return

    payload = llm.compare(REGION_DIFFERENCE_SYSTEM_PROMPT, build_region_difference_prompt(case.topic, evidence))
    by_label = {item.label: item for item in evidence}
    _score_citations(result, list(payload.get("citations", []))[:MAX_DIFFERENCE_CITATIONS], by_label)
    try:
        difference = parse_region_difference(payload, evidence)
    except ReasoningError as error:
        result.add("parses_with_valid_citations", False, detail=str(error))
        return
    result.add("parses_with_valid_citations", True)
    result.add("cites_both_regions", {c.region for c in difference.citations} == set(REGIONS),
               detail="a comparison must cite at least one EU and one UK source")
    result.add("states_key_difference", bool(difference.key_difference.strip()))
    _check_language(result, " ".join([difference.eu_summary, difference.uk_summary, difference.key_difference]))


def run_case(case: EvalCase, retriever: Retriever, llm: EvalLLM | None) -> CaseResult:
    """Run one eval case; with ``llm=None`` only the retrieval-level checks run.

    Any exception from retrieval or the model call aborts just this case and is
    recorded on the result, so one failure never hides the rest of the run.
    """
    result = CaseResult(case.id, case.kind)
    try:
        if case.kind == KIND_RECOMMENDATION:
            _run_recommendation(case, retriever, llm, result)
        elif case.kind == KIND_COMPARISON:
            _run_comparison(case, retriever, llm, result)
        else:
            raise ValueError(f"unknown case kind {case.kind!r}")
    except Exception as error:  # noqa: BLE001 - recorded, not swallowed
        result.error = f"{type(error).__name__}: {error}"
    return result


def run_evaluation(cases: list[EvalCase], retriever: Retriever, llm: EvalLLM | None, model: str = "") -> EvalReport:
    """Run every case and return the aggregate report (retrieval-only when ``llm`` is ``None``)."""
    results = [run_case(case, retriever, llm) for case in cases]
    return EvalReport(results, "full" if llm is not None else "retrieval-only", model)
