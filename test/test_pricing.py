from decimal import Decimal

import pytest

from subretrans.pricing import ModelPricing, calculate_cost, parse_model_pricing


def table() -> dict:
    return {
        "schema": "cchp.pricing-table/v1",
        "version": "abc123",
        "refreshed_at": "2026-09-20T00:00:00Z",
        "models": [
            {
                "slug": "google/gemini-test",
                "model_name": "gemini-test",
                "aliases": ["gemini/test"],
                "pricing": [
                    {
                        "provider": "proxy",
                        "official": False,
                        "charges": {
                            "prompt": {"unit": "per_M_tokens", "price": "1"},
                            "completion": {"unit": "per_M_tokens", "price": "2"},
                        },
                    },
                    {
                        "provider": "google",
                        "official": True,
                        "charges": {
                            "prompt": {"unit": "per_M_tokens", "price": "2"},
                            "completion": {"unit": "per_M_tokens", "price": "12"},
                        },
                    },
                ],
            }
        ],
    }


@pytest.mark.parametrize(
    "name", ("gemini-test", "google/gemini-test", "gemini/test")
)
def test_parse_model_pricing_matches_exact_names_and_prefers_official(name) -> None:
    result = parse_model_pricing(table(), name)

    assert result == ModelPricing(
        model_name="gemini-test",
        provider="google",
        version="abc123",
        refreshed_at="2026-09-20T00:00:00Z",
        prompt_per_million=Decimal("2"),
        completion_per_million=Decimal("12"),
    )


def test_parse_model_pricing_returns_none_for_unknown_model() -> None:
    assert parse_model_pricing(table(), "not-present") is None


def test_calculate_cost_uses_per_million_rates() -> None:
    pricing = parse_model_pricing(table(), "gemini-test")
    assert pricing is not None
    assert calculate_cost(
        pricing, prompt_tokens=500_000, completion_tokens=250_000
    ) == Decimal("4")


def test_parse_model_pricing_rejects_wrong_schema_and_units() -> None:
    payload = table()
    payload["schema"] = "other"
    with pytest.raises(ValueError, match="schema"):
        parse_model_pricing(payload, "gemini-test")

    payload = table()
    payload["models"][0]["pricing"][1]["charges"]["prompt"]["unit"] = "per_token"
    with pytest.raises(ValueError, match="per_M_tokens"):
        parse_model_pricing(payload, "gemini-test")
