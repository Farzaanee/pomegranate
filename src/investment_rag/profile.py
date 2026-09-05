"""User profile schema and the retrieval queries it implies."""

from dataclasses import dataclass

GOALS = ("retirement", "house_deposit", "general_growth", "emergency_fund")
RISK_TOLERANCES = ("low", "medium", "high")
REGIONS = ("EU", "UK")


@dataclass(frozen=True)
class UserProfile:
    """A user's investing situation, used to scope evidence retrieval and reasoning.

    ``monthly_income`` and ``investable_amount`` are in the user's local currency
    (GBP for UK, EUR for EU); they inform suitability framing only and are never
    treated as exact financial totals.
    """

    monthly_income: float
    investable_amount: float
    goal: str
    timeline_years: int
    risk_tolerance: str
    region: str

    def __post_init__(self) -> None:
        """Reject a profile with negative amounts or an unrecognized enum value."""
        if self.monthly_income < 0 or self.investable_amount < 0:
            raise ValueError("monthly_income and investable_amount must be non-negative.")
        if self.timeline_years <= 0:
            raise ValueError("timeline_years must be positive.")
        if self.goal not in GOALS:
            raise ValueError(f"goal must be one of {GOALS}, got {self.goal!r}.")
        if self.risk_tolerance not in RISK_TOLERANCES:
            raise ValueError(f"risk_tolerance must be one of {RISK_TOLERANCES}, got {self.risk_tolerance!r}.")
        if self.region not in REGIONS:
            raise ValueError(f"region must be one of {REGIONS}, got {self.region!r}.")

    def retrieval_queries(self) -> list[str]:
        """Build the sub-queries used to gather region-scoped supporting evidence.

        Each query targets one suitability dimension (horizon, risk, fees, goal,
        region-specific product) so the retrieved evidence set covers the whole
        profile rather than just its closest single match.
        """
        horizon = "short" if self.timeline_years < 3 else "medium" if self.timeline_years < 10 else "long"
        queries = [
            f"diversification and risk for a {horizon}-term investment horizon",
            f"{self.risk_tolerance} risk tolerance investment options",
            "fees and costs of investment products",
            f"investing for {self.goal.replace('_', ' ')}",
        ]
        queries.append("ISA versus general investment account" if self.region == "UK"
                        else "retail investor protections under MiFID II")
        return queries
