from __future__ import annotations

from pydantic import BaseModel

from guarded_agent.config import BudgetConfig
from guarded_agent.state import BudgetCounters


class BudgetVerdict(BaseModel):
    breached: bool
    reason: str | None = None


def check_budget(counters: BudgetCounters, limits: BudgetConfig) -> BudgetVerdict:
    """Compare usage counters against configured caps.

    Zero I/O: takes counters and limits as plain data, never reads a clock
    or makes a network call. Whoever updates `counters.elapsed_seconds` is
    responsible for actually reading the clock -- this function only
    compares numbers it's handed, so it's a pure function of state and
    testable without mocks.
    """
    if counters.steps_used >= limits.max_steps:
        return BudgetVerdict(breached=True, reason=f"max steps ({limits.max_steps}) reached")
    if counters.tool_calls_used >= limits.max_tool_calls:
        return BudgetVerdict(
            breached=True, reason=f"max tool calls ({limits.max_tool_calls}) reached"
        )
    if counters.tokens_used >= limits.max_tokens:
        return BudgetVerdict(breached=True, reason=f"max tokens ({limits.max_tokens}) reached")
    if counters.elapsed_seconds >= limits.max_wall_clock_seconds:
        return BudgetVerdict(
            breached=True,
            reason=f"max wall-clock time ({limits.max_wall_clock_seconds}s) reached",
        )
    return BudgetVerdict(breached=False)
