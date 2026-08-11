"""Estimate Gemini API costs from response token usage."""

from __future__ import annotations

from typing import Any


# Google Gemini Developer API Standard paid-tier rates, USD per 1M tokens.
MODEL_PRICING_USD_PER_MILLION = {
    "gemini-3.5-flash": {"input": 1.50, "output": 9.00},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
    "gemini-3.1-pro-preview": {"input": 2.00, "output": 12.00},
    "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    "gemini-2.5-flash-lite": {"input": 0.10, "audio_input": 0.30, "output": 0.40, "cached_input": 0.01},
    "gemini-3.6-flash": {"input": 1.50, "output": 7.50, "cached_input": 0.15},
}

PRICING_EFFECTIVE_DATE = "2026-08-11"


def estimate_cost(
    usage: dict[str, Any] | None,
    model: str,
    billing_tier: str = "paid",
    modality: str = "text",
) -> dict[str, float | int | str]:
    """Return a transparent token-based cost estimate."""
    usage = usage or {}
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    rates = MODEL_PRICING_USD_PER_MILLION.get(model, {"input": 0.0, "output": 0.0})
    cached_tokens = int(
        usage.get("cached_tokens", usage.get("cached_content_token_count", usage.get("total_cached_tokens", 0))) or 0
    )
    uncached_tokens = max(0, input_tokens - cached_tokens)
    input_rate = float(rates.get("audio_input") if modality == "audio" and rates.get("audio_input") is not None else rates.get("input", 0.0))
    cached_rate = float(rates.get("cached_input", input_rate))
    paid_input_cost = (uncached_tokens * input_rate + cached_tokens * cached_rate) / 1_000_000
    paid_output_cost = output_tokens * rates["output"] / 1_000_000
    paid_total = paid_input_cost + paid_output_cost
    actual_total = 0.0 if billing_tier == "free" else paid_total
    return {
        "model": model,
        "billing_tier": billing_tier,
        "input_tokens": input_tokens,
        "cached_tokens": cached_tokens,
        "output_tokens": output_tokens,
        "input_cost_usd": 0.0 if billing_tier == "free" else paid_input_cost,
        "output_cost_usd": 0.0 if billing_tier == "free" else paid_output_cost,
        "total_cost_usd": actual_total,
        "paid_equivalent_usd": paid_total,
        "pricing_effective_date": PRICING_EFFECTIVE_DATE,
    }


def estimate_video_plan_cost(
    *,
    duration_seconds: float,
    transcript_tokens: int,
    segment_count: int,
    hard_sentence_count: int,
    transcript_provider: str,
    billing_tier: str = "paid",
    media_resolution: str = "low",
) -> dict[str, Any]:
    """Estimate the planned video pipeline before its output exists."""
    video_tokens = 0
    if transcript_provider not in {"youtube_caption", "manual_caption"}:
        video_tokens = int(max(0, duration_seconds) * (100 if media_resolution == "low" else 300))
    batch_input = max(0, transcript_tokens) * 2 + max(1, segment_count) * 350
    expected_batch_output = max(1, segment_count) * 1800
    maximum_batch_output = max(1, segment_count) * 3000
    deep_input = max(0, hard_sentence_count) * 800
    expected_deep_output = max(0, hard_sentence_count) * 1200
    maximum_deep_output = max(0, hard_sentence_count) * 2000
    ingest = (
        estimate_cost(
            {"input_tokens": video_tokens, "output_tokens": transcript_tokens},
            "gemini-3.6-flash", billing_tier, modality="video",
        )
        if video_tokens else
        {"input_tokens": 0, "output_tokens": 0, "total_cost_usd": 0.0, "paid_equivalent_usd": 0.0}
    )
    batch_expected = estimate_cost(
        {"input_tokens": batch_input, "output_tokens": expected_batch_output},
        "gemini-3.5-flash-lite", billing_tier,
    )
    deep_expected = estimate_cost(
        {"input_tokens": deep_input, "output_tokens": expected_deep_output},
        "gemini-3.6-flash", billing_tier,
    )
    batch_max = estimate_cost(
        {"input_tokens": batch_input, "output_tokens": maximum_batch_output},
        "gemini-3.5-flash-lite", billing_tier,
    )
    deep_max = estimate_cost(
        {"input_tokens": deep_input, "output_tokens": maximum_deep_output},
        "gemini-3.6-flash", billing_tier,
    )
    expected = sum_costs([ingest, batch_expected, deep_expected])
    maximum = sum_costs([ingest, batch_max, deep_max])
    analysis_expected = sum_costs([batch_expected, deep_expected])
    analysis_maximum = sum_costs([batch_max, deep_max])
    return {
        "duration_seconds": duration_seconds,
        "transcript_tokens": transcript_tokens,
        "segment_count": segment_count,
        "batch_count": max(1, (segment_count + 3) // 4),
        "hard_sentence_count": hard_sentence_count,
        "video_input_tokens": video_tokens,
        "expected": expected,
        "maximum": maximum,
        "ingest": ingest,
        "analysis_expected": analysis_expected,
        "analysis_maximum": analysis_maximum,
        "pricing_effective_date": PRICING_EFFECTIVE_DATE,
    }


def sum_costs(costs: list[dict[str, Any]]) -> dict[str, float | int]:
    """Sum multiple cost estimates."""
    return {
        "input_tokens": sum(int(cost.get("input_tokens", 0)) for cost in costs),
        "output_tokens": sum(int(cost.get("output_tokens", 0)) for cost in costs),
        "total_cost_usd": sum(float(cost.get("total_cost_usd", 0)) for cost in costs),
        "paid_equivalent_usd": sum(float(cost.get("paid_equivalent_usd", 0)) for cost in costs),
    }


def estimate_run_costs(
    runs: list[dict[str, Any]] | None,
    fallback_usage: dict[str, Any] | None,
    fallback_model: str,
    billing_tier: str = "paid",
) -> dict[str, float | int]:
    """Price model-aware usage runs once, with a legacy aggregate fallback."""
    unique_runs = []
    seen_ids: set[str] = set()
    for index, run in enumerate(runs or []):
        run_id = str(run.get("run_id") or f"legacy-run-{index}")
        if run_id in seen_ids:
            continue
        seen_ids.add(run_id)
        unique_runs.append(run)
    if not unique_runs:
        return sum_costs([estimate_cost(fallback_usage, fallback_model, billing_tier)])
    return sum_costs(
        [
            estimate_cost(
                run.get("usage"),
                str(run.get("model_used") or fallback_model),
                billing_tier,
            )
            for run in unique_runs
        ]
    )


def format_cost(usd: float, usd_to_jpy: float = 155) -> str:
    """Format a small USD estimate with its approximate JPY equivalent."""
    return f"${usd:.6f} (~¥{usd * usd_to_jpy:,.0f} JPY)"


def budget_status(
    budget_jpy: float,
    spent_before_jpy: float,
    session_cost_usd: float,
    usd_to_jpy: float = 155,
) -> dict[str, float]:
    """Estimate remaining API budget in JPY from app-tracked costs."""
    budget = max(0.0, float(budget_jpy or 0))
    spent_before = max(0.0, float(spent_before_jpy or 0))
    session_spent = max(0.0, float(session_cost_usd or 0) * float(usd_to_jpy or 0))
    total_spent = spent_before + session_spent
    remaining = max(0.0, budget - total_spent)
    used_ratio = total_spent / budget if budget else 0.0
    remaining_ratio = remaining / budget if budget else 0.0
    return {
        "budget_jpy": budget,
        "spent_before_jpy": spent_before,
        "session_spent_jpy": session_spent,
        "total_spent_jpy": total_spent,
        "remaining_jpy": remaining,
        "used_ratio": used_ratio,
        "remaining_ratio": remaining_ratio,
    }
