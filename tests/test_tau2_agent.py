from __future__ import annotations

from guarded_agent.adapters.tau2_agent import TurnCost

# TurnCost fixes a real gap found while calibrating a live Haiku ablation
# run (PLAN.md commit 24 notes): policy_checker and critic LLM calls never
# went through tau2's own generate(), so their cost was silently invisible
# to every cost figure this project reports. A single turn can also call
# the main agent generate_fn twice (agent_revise, on a critic rejection) --
# TurnCost.add must sum every call this turn, not just remember the last one,
# and reset() must clear it between turns so costs don't compound.


def test_add_accumulates_across_multiple_calls() -> None:
    turn_cost = TurnCost()

    turn_cost.add(0.01, {"prompt_tokens": 100, "completion_tokens": 10})
    turn_cost.add(0.002, {"prompt_tokens": 20, "completion_tokens": 5})

    assert turn_cost.cost == 0.012
    assert turn_cost.usage == {"prompt_tokens": 120, "completion_tokens": 15}


def test_add_treats_none_cost_as_zero_not_as_poisoning_the_total() -> None:
    """A None cost from one call (e.g. an unpriced model) must not wipe out
    real cost already accumulated from other calls this turn -- unlike
    tau2's own get_cost(), which treats any None as poisoning the whole sum."""
    turn_cost = TurnCost()
    turn_cost.add(0.01, {"prompt_tokens": 100, "completion_tokens": 10})

    turn_cost.add(None, None)

    assert turn_cost.cost == 0.01


def test_add_with_no_usage_does_not_crash() -> None:
    turn_cost = TurnCost()

    turn_cost.add(0.005, None)

    assert turn_cost.cost == 0.005
    assert turn_cost.usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_reset_clears_accumulated_cost_and_usage() -> None:
    turn_cost = TurnCost()
    turn_cost.add(0.01, {"prompt_tokens": 100, "completion_tokens": 10})

    turn_cost.reset()

    assert turn_cost.cost == 0.0
    assert turn_cost.usage == {"prompt_tokens": 0, "completion_tokens": 0}


def test_cost_is_never_none_even_before_any_call() -> None:
    """Unlike the old LastGeneration (cost: float | None = None), a fresh
    TurnCost must start at a real 0.0 -- tau2's get_cost() treats any None
    cost on a message as poisoning the whole conversation's agent_cost to
    None (verified live, PLAN.md commit 21 v2 smoke run)."""
    turn_cost = TurnCost()

    assert turn_cost.cost == 0.0
    assert turn_cost.usage == {"prompt_tokens": 0, "completion_tokens": 0}
