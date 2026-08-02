from __future__ import annotations

import pytest

from guarded_agent.memory.case_store import CaseStore
from guarded_agent.memory.retention import redact_pii
from guarded_agent.memory.session import CaseRecord, build_case_record
from guarded_agent.state import AgentState, Message, ToolCall


def test_redact_pii_scrubs_email() -> None:
    result = redact_pii("Contact me at jane.doe@example.com about this.")
    assert "jane.doe@example.com" not in (result or "")
    assert "[REDACTED_EMAIL]" in (result or "")


def test_redact_pii_scrubs_phone_number() -> None:
    result = redact_pii("Call me at 555-123-4567 tomorrow.")
    assert "555-123-4567" not in (result or "")
    assert "[REDACTED_PHONE]" in (result or "")


def test_redact_pii_scrubs_card_number() -> None:
    result = redact_pii("My card is 4111 1111 1111 1111, please use it.")
    assert "4111 1111 1111 1111" not in (result or "")
    assert "[REDACTED_CARD]" in (result or "")


def test_redact_pii_leaves_ordinary_text_untouched() -> None:
    assert redact_pii("I'd like a refund on order 44821.") == "I'd like a refund on order 44821."


def test_redact_pii_passes_through_none() -> None:
    assert redact_pii(None) is None


def _state_with_pii(escalated: bool = False, escalation_reason: str | None = None) -> AgentState:
    state = AgentState(
        conversation=[
            Message(
                role="user", content="Hi, I'm jane.doe@example.com, order 12345 never arrived."
            ),
            Message(
                role="assistant",
                tool_calls=[
                    ToolCall(id="1", name="get_order_details", arguments={"order_id": "1"})
                ],
            ),
            Message(role="tool", content="found it", tool_call_id="1"),
            Message(role="assistant", content="Call me back at 555-987-6543 if that's wrong."),
        ]
    )
    if escalated:
        state = state.escalate(escalation_reason or "budget breached")
    return state


def test_build_case_record_excludes_pii_fields() -> None:
    """PLAN.md commit 19's literal acceptance check."""
    record = build_case_record(_state_with_pii(), session_id="session-1")

    assert "jane.doe@example.com" not in (record.issue_summary or "")
    assert "555-987-6543" not in (record.resolution_summary or "")
    assert "[REDACTED_EMAIL]" in (record.issue_summary or "")
    assert "[REDACTED_PHONE]" in (record.resolution_summary or "")


def test_build_case_record_marks_resolved_when_not_escalated() -> None:
    record = build_case_record(_state_with_pii(escalated=False), session_id="session-1")
    assert record.resolved is True
    assert record.escalation_reason is None


def test_build_case_record_marks_unresolved_when_escalated() -> None:
    record = build_case_record(
        _state_with_pii(escalated=True, escalation_reason="3 consecutive tool failures"),
        session_id="session-2",
    )
    assert record.resolved is False
    assert record.escalation_reason == "3 consecutive tool failures"


def test_build_case_record_collects_distinct_tool_names_only() -> None:
    state = AgentState(
        conversation=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="1", name="get_order_details", arguments={})],
            ),
            Message(role="tool", content="ok", tool_call_id="1"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="2", name="get_order_details", arguments={})],
            ),
            Message(role="tool", content="ok", tool_call_id="2"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="3", name="issue_refund", arguments={})],
            ),
        ]
    )
    record = build_case_record(state, session_id="session-3")
    assert record.tool_names_used == ["get_order_details", "issue_refund"]


def test_build_case_record_handles_empty_conversation() -> None:
    record = build_case_record(AgentState(), session_id="session-4")
    assert record.issue_summary is None
    assert record.resolution_summary is None
    assert record.tool_names_used == []


@pytest.fixture(scope="module")
def case_store() -> CaseStore:
    # Loads the real local embedding model once for the module -- no live API
    # calls (CLAUDE.md's testing convention), same pattern as
    # test_policy_retrieval.py's `retriever` fixture.
    return CaseStore()


def test_empty_case_store_retrieve_returns_nothing(case_store: CaseStore) -> None:
    assert case_store.retrieve("anything", k=3) == []


def test_case_store_round_trips_an_added_case(case_store: CaseStore) -> None:
    record = CaseRecord(
        session_id="session-refund",
        issue_summary="Customer wants a refund for a delayed order.",
        resolution_summary="Refund issued after confirming order status.",
        resolved=True,
        tool_names_used=["get_order_details", "issue_refund"],
    )
    case_store.add_case(record)

    results = case_store.retrieve("help with a refund for a late order", k=1)

    assert len(results) == 1
    assert results[0] == record


def test_case_store_retrieve_ignores_unrelated_cases(case_store: CaseStore) -> None:
    shipping_case = CaseRecord(
        session_id="session-shipping",
        issue_summary="Customer asked how long shipping takes.",
        resolution_summary="Explained standard shipping windows.",
        resolved=True,
        tool_names_used=[],
    )
    case_store.add_case(shipping_case)

    results = case_store.retrieve("help with a refund for a late order", k=1)

    assert len(results) == 1
    assert results[0].session_id == "session-refund"
