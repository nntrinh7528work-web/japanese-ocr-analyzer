from modules.cost_estimator import budget_status, estimate_cost, format_cost, sum_costs


def test_paid_cost_uses_gemini_flash_rates():
    cost = estimate_cost(
        {"input_tokens": 1_000_000, "output_tokens": 1_000_000},
        "gemini-2.5-flash",
        "paid",
    )
    assert cost["input_cost_usd"] == 0.30
    assert cost["output_cost_usd"] == 2.50
    assert cost["total_cost_usd"] == 2.80


def test_free_tier_is_zero_but_keeps_paid_equivalent():
    cost = estimate_cost({"input_tokens": 1000, "output_tokens": 1000}, "gemini-2.5-flash", "free")
    assert cost["total_cost_usd"] == 0
    assert cost["paid_equivalent_usd"] > 0


def test_sum_and_format_costs():
    costs = [
        estimate_cost({"input_tokens": 100, "output_tokens": 200}, "gemini-2.5-flash"),
        estimate_cost({"input_tokens": 300, "output_tokens": 400}, "gemini-2.5-flash"),
    ]
    total = sum_costs(costs)
    assert total["input_tokens"] == 400
    assert total["output_tokens"] == 600
    assert "JPY" in format_cost(total["total_cost_usd"])


def test_output_cost_can_include_thinking_tokens():
    cost = estimate_cost(
        {"input_tokens": 100, "candidate_tokens": 200, "thinking_tokens": 300, "output_tokens": 500},
        "gemini-2.5-flash",
    )
    assert cost["output_tokens"] == 500
    assert cost["output_cost_usd"] == 500 * 2.50 / 1_000_000


def test_budget_status_tracks_remaining_jpy():
    status = budget_status(100_000, 20_000, 1.0, 155)

    assert status["session_spent_jpy"] == 155
    assert status["total_spent_jpy"] == 20_155
    assert status["remaining_jpy"] == 79_845
    assert status["remaining_ratio"] == 0.79845
