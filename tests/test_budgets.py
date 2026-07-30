from __future__ import annotations

import pytest

from guarded_agent.config import BudgetConfig
from guarded_agent.guardrails.budgets import check_budget
from guarded_agent.state import BudgetCounters

LIMITS = BudgetConfig(max_steps=5, max_tool_calls=5, max_tokens=1000, max_wall_clock_seconds=60.0)


def test_under_all_caps_is_not_breached() -> None:
    counters = BudgetCounters(steps_used=1, tool_calls_used=1, tokens_used=10, elapsed_seconds=1.0)
    verdict = check_budget(counters, LIMITS)
    assert verdict.breached is False
    assert verdict.reason is None


@pytest.mark.parametrize(
    ("counters", "expected_substring"),
    [
        (BudgetCounters(steps_used=5), "steps"),
        (BudgetCounters(tool_calls_used=5), "tool calls"),
        (BudgetCounters(tokens_used=1000), "tokens"),
        (BudgetCounters(elapsed_seconds=60.0), "wall-clock"),
    ],
)
def test_each_cap_at_the_limit_is_breached(
    counters: BudgetCounters, expected_substring: str
) -> None:
    verdict = check_budget(counters, LIMITS)
    assert verdict.breached is True
    assert verdict.reason is not None
    assert expected_substring in verdict.reason


def test_one_cap_over_limit_is_breached_even_if_others_are_fine() -> None:
    counters = BudgetCounters(steps_used=999, tool_calls_used=0, tokens_used=0)
    verdict = check_budget(counters, LIMITS)
    assert verdict.breached is True
