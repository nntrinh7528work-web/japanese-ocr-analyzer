"""Estimate Gemini API costs from response token usage."""

from __future__ import annotations

from typing import Any


# Google Gemini Developer API Standard paid-tier rates, USD per 1M tokens.
MODEL_PRICING_USD_PER_MILLION = {
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-pro": {"input": 2.25, "output": 18.00},
}


def estimate_cost(
    usage: dict[str, Any] | None,
    model: str,
    billing_tier: str = "paid",
) -> dict[str, float | int | str]:
    """Return a transparent token-based cost estimate."""
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    rates = MODEL_PRICING_USD_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    paid_input_cost = input_tokens * rates["input"] / 1_000_000
    paid_output_cost = output_tokens * rates["output"] / 1_000_000
    paid_total = paid_input_cost + paid_output_cost
    actual_total = 0.0 if billing_tier == "free" else paid_total
    return {
        "model": model,
        "billing_tier": billing_tier,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": 0.0 if billing_tier == "free" else paid_input_cost,
        "output_cost_usd": 0.0 if billing_tier == "free" else paid_output_cost,
        "total_cost_usd": actual_total,
        "paid_equivalent_usd": paid_total,
    }


def sum_costs(costs: list[dict[str, Any]]) -> dict[str, float | int]:
    """Sum multiple cost estimates."""
    return {
        "input_tokens": sum(int(cost.get("input_tokens", 0)) for cost in costs),
        "output_tokens": sum(int(cost.get("output_tokens", 0)) for cost in costs),
        "total_cost_usd": sum(float(cost.get("total_cost_usd", 0)) for cost in costs),
        "paid_equivalent_usd": sum(float(cost.get("paid_equivalent_usd", 0)) for cost in costs),
    }


def format_cost(usd: float, usd_to_vnd: float = 25_500) -> str:
    """Format a small USD estimate with its approximate VND equivalent."""
    return f"${usd:.6f} (~{usd * usd_to_vnd:,.0f} VND)"

