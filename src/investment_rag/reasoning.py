"""Agentic reasoning layer: combines a user profile with retrieved evidence to
produce a citation-backed recommendation via structured LLM output."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from .models import Chunk
from .profile import UserProfile
from .retrieval import Retriever

if TYPE_CHECKING:
    import anthropic

DISCLAIMER = (
    "Educational only, not regulated financial advice. This tool explains general "
    "investing principles grounded in official public sources; it does not "
    "recommend specific products and is not a substitute for professional advice."
)

SYSTEM_PROMPT = f"""You are an educational investment-literacy assistant covering the EU and UK.

{DISCLAIMER}

You will be given a user's profile and a numbered set of evidence passages
retrieved from official sources (ESMA, MoneyHelper, FCA). Ground every reasoning
step in those passages only — never introduce outside facts, specific products,
or numeric return projections. Explain your reasoning in plain, jargon-free
language a non-expert can follow, and note any caveats (fees, existing debt,
hype, risk) the user should weigh before acting. Cite only passage labels that
were given to you."""

RECOMMENDATION_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
        "considerations": {"type": "array", "items": {"type": "string"}},
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
    "required": ["summary", "reasoning_steps", "considerations", "citations"],
    "additionalProperties": False,
}


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
class Recommendation:
    """A grounded, plain-language recommendation produced from a user profile."""

    summary: str
    reasoning_steps: list[str]
    considerations: list[str]
    citations: list[Citation]


class RecommendationLLM(Protocol):
    """Produces a structured recommendation payload from rendered prompts."""

    def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Return a dict matching ``RECOMMENDATION_SCHEMA`` for the given prompts."""
        ...


class ClaudeRecommendationLLM:
    """Calls Claude with a JSON-schema output constraint for the recommendation.

    Defaults to ``claude-opus-5`` for reasoning quality; pass ``model`` to use a
    cheaper model such as ``claude-sonnet-5`` if you want to trade quality for
    cost — that trade-off is the deployer's call, not this class's default.
    """

    def __init__(self, model: str = "claude-opus-5", client: anthropic.Anthropic | None = None) -> None:
        """Create an Anthropic client, reading credentials from the environment."""
        import anthropic

        self.model = model
        self._client = client or anthropic.Anthropic()

    def recommend(self, system_prompt: str, user_prompt: str) -> dict[str, object]:
        """Call Claude and parse its schema-constrained JSON response."""
        response = self._client.messages.create(
            model=self.model,
            max_tokens=8000,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            output_config={"format": {"type": "json_schema", "schema": RECOMMENDATION_SCHEMA}},
        )
        text = next(block.text for block in response.content if block.type == "text")
        return json.loads(text)


def gather_evidence(retriever: Retriever, profile: UserProfile, per_query_limit: int = 3) -> list[EvidencePassage]:
    """Run the profile's derived queries and return deduped, sequentially labeled passages."""
    seen: dict[str, EvidencePassage] = {}
    for query in profile.retrieval_queries():
        for result in retriever.search(query, limit=per_query_limit, region=profile.region):
            if result.chunk.id not in seen:
                seen[result.chunk.id] = EvidencePassage(str(len(seen) + 1), result.chunk)
    return list(seen.values())


def build_user_prompt(profile: UserProfile, evidence: list[EvidencePassage]) -> str:
    """Render the profile and numbered evidence passages into the user turn."""
    passages = "\n\n".join(
        f"[{item.label}] {item.chunk.source_name} ({item.chunk.region}): {item.chunk.text}"
        for item in evidence
    )
    return (
        "User profile:\n"
        f"- Region: {profile.region}\n"
        f"- Goal: {profile.goal}\n"
        f"- Timeline: {profile.timeline_years} years\n"
        f"- Risk tolerance: {profile.risk_tolerance}\n"
        f"- Monthly income: {profile.monthly_income}\n"
        f"- Investable amount: {profile.investable_amount}\n\n"
        f"Evidence passages:\n{passages}\n\n"
        "Every reasoning step must be traceable to at least one passage label above; "
        "cite only labels that appear above."
    )


def parse_recommendation(payload: dict[str, object], evidence: list[EvidencePassage]) -> Recommendation:
    """Validate the LLM's payload and resolve citations against real evidence.

    A citation whose label doesn't match a retrieved passage is dropped rather
    than trusted, since the model can invent labels; if none survive, the
    recommendation isn't grounded and this raises instead of returning it.
    """
    by_label = {item.label: item.chunk for item in evidence}
    citations = []
    for raw in payload.get("citations", []):
        chunk = by_label.get(raw.get("label"))
        if chunk is not None:
            citations.append(Citation(raw["label"], chunk.source_name, chunk.region, chunk.url, chunk.title,
                                       raw.get("quote", "")))
    if not citations:
        raise ReasoningError("The model's response cited no valid evidence passages.")
    return Recommendation(
        summary=payload["summary"],
        reasoning_steps=list(payload.get("reasoning_steps", [])),
        considerations=list(payload.get("considerations", [])),
        citations=citations,
    )


class ReasoningAgent:
    """Combines a user profile with retrieved evidence to produce a grounded recommendation."""

    def __init__(self, retriever: Retriever, llm: RecommendationLLM, per_query_limit: int = 3) -> None:
        """Wire the retriever and LLM this agent calls for each request."""
        self.retriever = retriever
        self.llm = llm
        self.per_query_limit = per_query_limit

    def run(self, profile: UserProfile) -> Recommendation:
        """Gather region-scoped evidence, prompt the LLM, and return a cited recommendation."""
        evidence = gather_evidence(self.retriever, profile, self.per_query_limit)
        if not evidence:
            raise ReasoningError("No evidence retrieved for this profile's region; cannot ground a recommendation.")
        payload = self.llm.recommend(SYSTEM_PROMPT, build_user_prompt(profile, evidence))
        return parse_recommendation(payload, evidence)
