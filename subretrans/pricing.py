"""Claude Code Hub model-pricing lookup and best-effort cost estimation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

import requests

from .stats import UsageStats


logger = logging.getLogger(__name__)

CCH_PRICING_URL = "https://cch-plus.com/pricing/v1/models.json"
CCH_PRICING_SCHEMA = "cchp.pricing-table/v1"


@dataclass(frozen=True)
class ModelPricing:
    model_name: str
    provider: str
    version: str
    refreshed_at: str
    prompt_per_million: Decimal
    completion_per_million: Decimal


@dataclass(frozen=True)
class CostEstimate:
    pricing: ModelPricing
    cost: Decimal


def _required_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"CCH pricing {field} must be a non-empty string")
    return value


def _per_million_charge(charges: Any, name: str) -> Decimal:
    if not isinstance(charges, dict) or not isinstance(charges.get(name), dict):
        raise ValueError(f"CCH pricing variant has no {name} charge")
    charge = charges[name]
    if charge.get("unit") != "per_M_tokens":
        raise ValueError(f"CCH pricing {name} charge must use per_M_tokens")
    try:
        price = Decimal(charge["price"])
    except (InvalidOperation, KeyError, TypeError) as exc:
        raise ValueError(f"CCH pricing {name} charge has an invalid price") from exc
    if price < 0:
        raise ValueError(f"CCH pricing {name} charge must be non-negative")
    return price


def parse_model_pricing(payload: Any, model_name: str) -> ModelPricing | None:
    """Return an exact CCH model match, or ``None`` when the table lacks one."""

    if not isinstance(payload, dict) or payload.get("schema") != CCH_PRICING_SCHEMA:
        raise ValueError(f"CCH pricing schema must be {CCH_PRICING_SCHEMA}")
    version = _required_string(payload.get("version"), "version")
    refreshed_at = _required_string(payload.get("refreshed_at"), "refreshed_at")
    models = payload.get("models")
    if not isinstance(models, list):
        raise ValueError("CCH pricing models must be a list")

    for model in models:
        if not isinstance(model, dict):
            raise ValueError("CCH pricing model entries must be mappings")
        aliases = model.get("aliases", [])
        if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
            raise ValueError("CCH pricing model aliases must be strings")
        names = {model.get("model_name"), model.get("slug"), *aliases}
        if model_name not in names:
            continue

        variants = model.get("pricing")
        if not isinstance(variants, list) or not variants:
            raise ValueError(f"CCH pricing model {model_name} has no variants")
        variant = next(
            (item for item in variants if isinstance(item, dict) and item.get("official") is True),
            variants[0],
        )
        if not isinstance(variant, dict):
            raise ValueError(f"CCH pricing model {model_name} has an invalid variant")
        return ModelPricing(
            model_name=_required_string(model.get("model_name"), "model_name"),
            provider=_required_string(variant.get("provider"), "provider"),
            version=version,
            refreshed_at=refreshed_at,
            prompt_per_million=_per_million_charge(variant.get("charges"), "prompt"),
            completion_per_million=_per_million_charge(variant.get("charges"), "completion"),
        )
    return None


def load_model_pricing(model_name: str) -> ModelPricing | None:
    """Fetch the current CCH table and resolve an exact model match."""

    response = requests.get(CCH_PRICING_URL, timeout=30)
    response.raise_for_status()
    return parse_model_pricing(response.json(), model_name)


def calculate_cost(
    pricing: ModelPricing, *, prompt_tokens: int, completion_tokens: int
) -> Decimal:
    """Calculate USD cost from CCH per-million-token charges."""

    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("token counts must be non-negative")
    million = Decimal(1_000_000)
    return (
        Decimal(prompt_tokens) * pricing.prompt_per_million
        + Decimal(completion_tokens) * pricing.completion_per_million
    ) / million


def estimate_cost(model_name: str, usage: UsageStats) -> CostEstimate | None:
    """Best-effort cost report: any lookup failure is logged and yields ``None``."""

    try:
        pricing = load_model_pricing(model_name)
    except Exception as error:
        logger.warning("CCH pricing unavailable for %s: %s", model_name, error)
        return None
    if pricing is None:
        logger.info("CCH pricing has no exact match for %s", model_name)
        return None
    cost = calculate_cost(
        pricing, prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens
    )
    return CostEstimate(pricing, cost)
