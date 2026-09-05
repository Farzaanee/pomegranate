import pytest

from investment_rag.profile import UserProfile


def test_valid_profile_builds_region_appropriate_queries() -> None:
    profile = UserProfile(3000, 5000, "retirement", 20, "medium", "UK")

    queries = profile.retrieval_queries()

    assert any("ISA" in query for query in queries)
    assert any("long-term" in query for query in queries)
    assert any("medium risk" in query for query in queries)


def test_eu_profile_asks_about_mifid_instead_of_isa() -> None:
    profile = UserProfile(2000, 1000, "general_growth", 2, "low", "EU")

    queries = profile.retrieval_queries()

    assert any("MiFID" in query for query in queries)
    assert not any("ISA" in query for query in queries)
    assert any("short-term" in query for query in queries)


@pytest.mark.parametrize("kwargs", [
    {"monthly_income": -1},
    {"investable_amount": -1},
    {"timeline_years": 0},
    {"goal": "yacht"},
    {"risk_tolerance": "extreme"},
    {"region": "US"},
])
def test_invalid_profile_fields_are_rejected(kwargs) -> None:
    base = dict(monthly_income=2000, investable_amount=1000, goal="retirement",
                timeline_years=10, risk_tolerance="medium", region="UK")
    base.update(kwargs)

    with pytest.raises(ValueError):
        UserProfile(**base)
