"""Agentic reasoning layer: combines a user profile with retrieved evidence to
produce a citation-backed recommendation via structured LLM output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .models import Chunk
from .profile import REGIONS, UserProfile
from .retrieval import Retriever

if TYPE_CHECKING:
    import anthropic

DISCLAIMER = (
    "This is general information, not regulated financial advice. This tool explains "
    "general investing principles grounded in official public sources; it does not "
    "recommend specific products and is not a substitute for professional advice."
)

SYSTEM_PROMPT = f"""You are the reasoning step of an educational investment-literacy demo
covering the EU and UK. Every profile you receive is synthetic test data for a
portfolio project — there is no real person, no real money, and no account will
ever be opened from your output.

Your job is suitability education, not instructions: given a synthetic profile,
explain which *types* of investment vehicle (asset classes and account wrappers,
e.g. index funds, bonds, cash savings, an ISA wrapper) generally fit a profile
like this one and why, and explain the tradeoffs in plain language. Describe
categories and general principles. Never name a specific fund, ticker, or
provider; never state an amount to invest; never project returns; never phrase
anything as an order to act.

Ground every claim in the numbered evidence passages you are given, which are
retrieved from official sources (ESMA, MoneyHelper, FCA). Do not introduce
outside facts, and cite only passage labels that appear in the prompt. Write in
jargon-free language a non-expert can follow, list the real risks and the
caveats (fees, existing debt, emergency savings, hype) a reader should weigh,
and include this line verbatim as the final caveat:

"{DISCLAIMER}"
Be concise: at most 4 suitable options, 5 reasoning steps, 5 risks, 5 caveats,
and 8 citations.
"""

# Caps keep the model's completion (and thus latency) bounded; raise if outputs feel truncated.
MAX_SUITABLE_OPTIONS = 4
MAX_REASONING_STEPS = 5
MAX_RISKS = 5
MAX_CAVEATS = 5
MAX_CITATIONS = 8

# Anthropic's json_schema output format rejects maxItems, so counts are capped in the
# system prompt and enforced again by slicing in parse_recommendation.
RECOMMENDATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "suitable_options": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "vehicle_type": {"type": "string"},
                    "why_it_fits": {"type": "string"},
                    "tradeoffs": {"type": "string"},
                    "citation_labels": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["vehicle_type", "why_it_fits", "tradeoffs", "citation_labels"],
                "additionalProperties": False,
            },
        },
        "reasoning": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "caveats": {"type": "array", "items": {"type": "string"}},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["label", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["summary", "suitable_options", "reasoning", "risks", "caveats", "citations"],
    "additionalProperties": False,
}


REGION_DIFFERENCE_SYSTEM_PROMPT = f"""You are the multi-jurisdiction comparison step of the same
educational investment-literacy demo covering the EU and UK described above: explaining general
principles from official sources, never naming a product, an amount, or an instruction to act.

Given a topic and evidence passages labeled either "EU-N" or "UK-N", explain what generally
applies for an EU investor, what generally applies for a UK investor, and then state the single
most important difference between the two regions on this topic. An "EU-" labeled passage comes
only from EU sources (ESMA); a "UK-" labeled passage comes only from UK sources (FCA,
MoneyHelper). Never attribute an EU-labeled passage to a UK claim, or a UK-labeled passage to an
EU claim, and cite only labels that appear in the prompt. Write in jargon-free language a
non-expert can follow, and keep each summary to a few sentences.
"""

DIFFERENCE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "eu_summary": {"type": "string"},
        "uk_summary": {"type": "string"},
        "key_difference": {"type": "string"},
        "citations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "quote": {"type": "string"},
                },
                "required": ["label", "quote"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["eu_summary", "uk_summary", "key_difference", "citations"],
    "additionalProperties": False,
}

# Caps the evidence sent to the LLM (per region) so prompt size stays bounded.
MAX_COMPARISON_PASSAGES_PER_REGION = 4
MAX_DIFFERENCE_CITATIONS = 6


class ReasoningError(RuntimeError):
    """The reasoning layer could not produce a grounded recommendation."""


@dataclass(frozen=True)
class EvidencePassage:
    """A retrieved chunk labeled for citation (e.g. ``"1"``) within one request."""

    label: str
    chunk: Chunk


@dataclass(frozen=True)
class Citation:
    """One recommendation claim traced back to a specific retrieved passage."""

    label: str
    source_name: str
    region: str
    url: str
    title: str
    quote: str


@dataclass(frozen=True)
class SuitableOption:
    """One category of investment vehicle assessed against a profile, with its tradeoffs."""

    vehicle_type: str
    why_it_fits: str
    tradeoffs: str
    citation_labels: list[str]


@dataclass(frozen=True)
class Recommendation:
    """A grounded, plain-language suitability analysis produced from a user profile."""

    summary: str
    suitable_options: list[SuitableOption]
    reasoning: list[str]
    risks: list[str]
    caveats: list[str]
    citations: list[Citation]


@dataclass(frozen=True)
class RegionDifference:
    """A grounded, plain-language explanation of an EU/UK difference on one topic."""

    eu_summary: str
    uk_summary: str
    key_difference: str
    citations: list[Citation]


class RecommendationLLM(Protocol):
    """Produces a structured recommendation payload from rendered prompts."""

    def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Return a dict matching ``RECOMMENDATION_SCHEMA`` for the given prompts."""
        ...


class RegionDifferenceLLM(Protocol):
    """Produces a structured EU/UK comparison payload from rendered prompts."""

    def compare(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Return a dict matching ``DIFFERENCE_SCHEMA`` for the given prompts."""
        ...


class ClaudeRecommendationLLM:
    """Calls Claude with a JSON-schema output constraint for the recommendation.

    Defaults to ``claude-sonnet-5`` for latency; pass ``model`` to use
    ``claude-opus-5`` if you want to trade speed for reasoning quality — that
    trade-off is the deployer's call, not this class's default.
    """

    def __init__(
        self,
        model: str = "claude-sonnet-5",
        client: anthropic.Anthropic | None = None,
        api_key: str | None = None,
    ) -> None:
        """Create an Anthropic client.

        ``api_key`` is passed straight to the SDK; if omitted, the SDK falls
        back to the ``ANTHROPIC_API_KEY`` environment variable itself.
        """
        import anthropic

        self.model = model
        self._client = client or anthropic.Anthropic(api_key=api_key)

    def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Call Claude and parse its schema-constrained JSON response."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=4000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": RECOMMENDATION_SCHEMA}},
        )
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)

    def compare(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Call Claude and parse its schema-constrained EU/UK comparison response."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=2000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": DIFFERENCE_SCHEMA}},
        )
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


# Caps the evidence sent to the LLM so prompt size (and prefill time) stays bounded.
MAX_EVIDENCE_PASSAGES = 10
# Chunk text is truncated in the prompt only; citations still show the full quote to the user.
MAX_PASSAGE_CHARS = 600


def gather_evidence(retriever: Retriever, profile: UserProfile, per_query_limit: int = 3) -> list[EvidencePassage]:
    """Run the profile's derived queries and return deduped, sequentially labeled passages.

    Every result is re-checked against ``profile.region`` before being kept: the
    retriever is already asked to filter by region, but this routing step never
    trusts that filter alone, so a bug or a misconfigured store can't leak an
    out-of-region source into a recommendation.
    """
    seen: dict[str, EvidencePassage] = {}
    for query in profile.retrieval_queries():
        if len(seen) >= MAX_EVIDENCE_PASSAGES:
            break
        for result in retriever.search(query, limit=per_query_limit, region=profile.region):
            if result.chunk.region != profile.region:
                continue
            if result.chunk.id not in seen:
                seen[result.chunk.id] = EvidencePassage(str(len(seen) + 1), result.chunk)
            if len(seen) >= MAX_EVIDENCE_PASSAGES:
                break
    return list(seen.values())


def gather_region_comparison_evidence(
    retriever: Retriever, topic: str, per_region_limit: int = MAX_COMPARISON_PASSAGES_PER_REGION
) -> list[EvidencePassage]:
    """Run ``topic`` once per region and return passages labeled ``"EU-1"``, ``"UK-1"``, ….

    Labeling by region (rather than a shared running counter) keeps the region
    a passage came from legible in the prompt itself, so the model — and
    ``parse_region_difference``'s validation — can never confuse which region a
    citation belongs to.
    """
    passages: list[EvidencePassage] = []
    for region in REGIONS:
        for result in retriever.search(topic, limit=per_region_limit, region=region):
            if result.chunk.region != region:
                continue
            index = sum(1 for item in passages if item.chunk.region == region) + 1
            passages.append(EvidencePassage(f"{region}-{index}", result.chunk))
    return passages


def _truncate(text: str, max_chars: int) -> str:
    """Cut ``text`` to ``max_chars``, breaking on a word boundary where possible."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(" ", 1)[0] + "…"


def build_user_prompt(profile: UserProfile, evidence: list[EvidencePassage]) -> str:
    """Render the synthetic profile and numbered evidence passages into the user turn."""
    passages = "\n\n".join(
        f"[{item.label}] {item.chunk.source_name} ({item.chunk.region}): "
        f"{_truncate(item.chunk.text, MAX_PASSAGE_CHARS)}"
        for item in evidence
    )
    return (
        "Synthetic profile (generated test data, not a real person):\n"
        f"- Region: {profile.region}\n"
        f"- Goal: {profile.goal}\n"
        f"- Timeline: {profile.timeline_years} years\n"
        f"- Risk tolerance: {profile.risk_tolerance}\n"
        f"- Monthly income: {profile.monthly_income}\n"
        f"- Investable amount: {profile.investable_amount}\n\n"
        f"Evidence passages:\n{passages}\n\n"
        "Given this profile, identify which types of investment vehicle are generally "
        "suitable, explain in plain language why each one fits this profile and what its "
        "tradeoffs are, and cite the passages that support each point. Do not pick a single "
        "option, name a product or provider, or suggest an amount to invest. "
        "Every reasoning step must be traceable to at least one passage label above; "
        "cite only labels that appear above."
    )


def build_region_difference_prompt(topic: str, evidence: list[EvidencePassage]) -> str:
    """Render the topic and region-labeled evidence passages into the user turn."""
    passages = "\n\n".join(
        f"[{item.label}] {item.chunk.source_name} ({item.chunk.region}): "
        f"{_truncate(item.chunk.text, MAX_PASSAGE_CHARS)}"
        for item in evidence
    )
    return (
        f"Topic: {topic}\n\n"
        f"Evidence passages:\n{passages}\n\n"
        "Explain what generally applies for an EU investor and what generally applies for a UK "
        "investor on this topic, then state the single most important difference between the two "
        "regions. Cite only the passage labels above, and never cite an EU-labeled passage to "
        "support a UK claim or vice versa."
    )


def parse_region_difference(payload: dict[str, object], evidence: list[EvidencePassage]) -> RegionDifference:
    """Validate the LLM's comparison payload and resolve citations against real, region-matched evidence.

    Beyond dropping citation labels that don't match a retrieved passage (as
    ``parse_recommendation`` does), this also rejects any citation whose label
    prefix (``"EU-"`` / ``"UK-"``) doesn't match the region of the passage it
    resolved to — the model can't launder an EU claim through a UK-looking
    label or vice versa.
    """
    by_label = {item.label: item.chunk for item in evidence}
    citations = []
    for raw in payload.get("citations", [])[:MAX_DIFFERENCE_CITATIONS]:
        label = raw.get("label")
        chunk = by_label.get(label)
        if chunk is None:
            continue
        if not isinstance(label, str) or not label.startswith(f"{chunk.region}-"):
            continue
        citations.append(Citation(label, chunk.source_name, chunk.region, chunk.url, chunk.title,
                                   raw.get("quote", "")))
    if not citations:
        raise ReasoningError("The model's response cited no valid, region-matched evidence passages.")
    return RegionDifference(
        eu_summary=payload.get("eu_summary", ""),
        uk_summary=payload.get("uk_summary", ""),
        key_difference=payload.get("key_difference", ""),
        citations=citations,
    )


def explain_region_difference(
    retriever: Retriever, llm: RegionDifferenceLLM, topic: str,
    per_region_limit: int = MAX_COMPARISON_PASSAGES_PER_REGION,
) -> RegionDifference:
    """Gather region-labeled evidence for ``topic`` from both regions and ask for a grounded comparison."""
    evidence = gather_region_comparison_evidence(retriever, topic, per_region_limit)
    if not evidence:
        raise ReasoningError("No evidence retrieved for either region; cannot ground a comparison.")
    payload = llm.compare(REGION_DIFFERENCE_SYSTEM_PROMPT, build_region_difference_prompt(topic, evidence))
    return parse_region_difference(payload, evidence)


def _resolve_options(payload: dict[str, object], valid_labels: set[str]) -> list[SuitableOption]:
    """Build the suitable-option list, keeping only citation labels that were retrieved."""
    return [
        SuitableOption(
            vehicle_type=raw.get("vehicle_type", ""),
            why_it_fits=raw.get("why_it_fits", ""),
            tradeoffs=raw.get("tradeoffs", ""),
            citation_labels=[label for label in raw.get("citation_labels", []) if label in valid_labels],
        )
        for raw in payload.get("suitable_options", [])[:MAX_SUITABLE_OPTIONS]
    ]


def parse_recommendation(payload: dict[str, object], evidence: list[EvidencePassage]) -> Recommendation:
    """Validate the LLM's payload and resolve citations against real evidence.

    A citation whose label doesn't match a retrieved passage is dropped rather
    than trusted, since the model can invent labels; if none survive, the
    recommendation isn't grounded and this raises instead of returning it. The
    standing disclaimer is appended locally so the user always sees it even if
    the model omits it.
    """
    by_label = {item.label: item.chunk for item in evidence}
    citations = []
    for raw in payload.get("citations", [])[:MAX_CITATIONS]:
        chunk = by_label.get(raw.get("label"))
        if chunk is not None:
            citations.append(Citation(raw["label"], chunk.source_name, chunk.region, chunk.url, chunk.title,
                                       raw.get("quote", "")))
    if not citations:
        raise ReasoningError("The model's response cited no valid evidence passages.")
    caveats = [caveat for caveat in payload.get("caveats", []) if caveat != DISCLAIMER]
    return Recommendation(
        summary=payload["summary"],
        suitable_options=_resolve_options(payload, set(by_label)),
        reasoning=list(payload.get("reasoning", []))[:MAX_REASONING_STEPS],
        risks=list(payload.get("risks", []))[:MAX_RISKS],
        caveats=[*caveats[: MAX_CAVEATS - 1], DISCLAIMER],
        citations=citations,
    )


class ReasoningAgent:
    """Combines a user profile with retrieved evidence to produce a grounded suitability analysis."""

    def __init__(self, retriever: Retriever, llm: RecommendationLLM, per_query_limit: int = 3) -> None:
        """Wire the retriever and LLM this agent calls for each request."""
        self.retriever = retriever
        self.llm = llm
        self.per_query_limit = per_query_limit

    def run(self, profile: UserProfile) -> Recommendation:
        """Gather region-scoped evidence, prompt the LLM, and return a cited analysis."""
        evidence = gather_evidence(self.retriever, profile, self.per_query_limit)
        if not evidence:
            raise ReasoningError("No evidence retrieved for this profile's region; cannot ground a recommendation.")
        payload = self.llm.recommend(SYSTEM_PROMPT, build_user_prompt(profile, evidence))
        return parse_recommendation(payload, evidence)
