from __future__ import annotations

import pytest
from pydantic import ValidationError

from guarded_agent.state import (
    AgentState,
    BudgetCounters,
    Message,
    PolicyVerdict,
    ProposedAction,
    ToolCall,
)


def test_default_state_is_empty() -> None:
    state = AgentState()

    assert state.conversation == []
    assert state.plan is None
    assert state.proposed_action is None
    assert state.policy_verdict is None
    assert state.budget == BudgetCounters()
    assert state.escalated is False
    assert state.escalation_reason is None


def test_add_message_appends_without_mutating_original() -> None:
    state = AgentState()
    greeting = Message(role="assistant", content="Hi! How can I help you today?")

    updated = state.add_message(greeting)

    assert updated.conversation == [greeting]
    assert state.conversation == []  # original untouched


def test_add_message_preserves_order_across_multiple_calls() -> None:
    state = AgentState()
    first = Message(role="assistant", content="Hi!")
    second = Message(role="user", content="I need a refund.")

    state = state.add_message(first).add_message(second)

    assert [m.content for m in state.conversation] == ["Hi!", "I need a refund."]


def test_budget_increment_accumulates_without_mutating_original() -> None:
    budget = BudgetCounters()

    once = budget.increment(steps=1, tool_calls=1, tokens=500)
    twice = once.increment(steps=1, tokens=300)

    assert budget == BudgetCounters()  # original untouched
    assert once == BudgetCounters(steps_used=1, tool_calls_used=1, tokens_used=500)
    assert twice == BudgetCounters(steps_used=2, tool_calls_used=1, tokens_used=800)


def test_escalate_sets_flag_and_reason_without_mutating_original() -> None:
    state = AgentState()

    escalated = state.escalate("budget exceeded")

    assert escalated.escalated is True
    assert escalated.escalation_reason == "budget exceeded"
    assert state.escalated is False  # original untouched
    assert state.escalation_reason is None


def test_policy_verdict_rejects_invalid_verdict_value() -> None:
    with pytest.raises(ValidationError):
        PolicyVerdict(verdict="MAYBE", clause_id="R-12", reason="not a real verdict")  # type: ignore[arg-type]


def test_full_state_round_trips_through_json_serialization() -> None:
    state = AgentState(
        conversation=[
            Message(role="user", content="Can I get a refund?"),
            Message(
                role="assistant",
                content=None,
                tool_calls=[ToolCall(name="get_order_details", arguments={"order_id": "123"})],
            ),
        ],
        plan="Look up the order, then check the refund policy before acting.",
        proposed_action=ProposedAction(
            tool_name="issue_refund", arguments={"order_id": "123", "amount": 42.0}
        ),
        policy_verdict=PolicyVerdict(
            verdict="NEEDS_CONFIRMATION",
            clause_id="R-7",
            reason="Refund exceeds auto-approve limit.",
        ),
        budget=BudgetCounters(steps_used=3, tool_calls_used=1, tokens_used=1200),
        escalated=False,
    )

    restored = AgentState.model_validate_json(state.model_dump_json())

    assert restored == state
