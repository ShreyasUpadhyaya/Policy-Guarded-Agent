from __future__ import annotations

import pytest

from guarded_agent.guardrails.write_gate import (
    DispatchCache,
    compute_action_hash,
    evaluate_write_gate,
    is_affirmative,
)
from guarded_agent.state import Message, ProposedAction
from guarded_agent.tools.registry import ToolResult

ACTION = ProposedAction(
    id="call_1", tool_name="cancel_pending_order", arguments={"order_id": "#W1"}
)
OTHER_ACTION = ProposedAction(
    id="call_2", tool_name="cancel_pending_order", arguments={"order_id": "#W2"}
)


# --- compute_action_hash ----------------------------------------------------


def test_action_hash_is_deterministic() -> None:
    assert compute_action_hash(ACTION) == compute_action_hash(ACTION.model_copy())


def test_action_hash_ignores_argument_order() -> None:
    a = ProposedAction(tool_name="x", arguments={"a": 1, "b": 2})
    b = ProposedAction(tool_name="x", arguments={"b": 2, "a": 1})
    assert compute_action_hash(a) == compute_action_hash(b)


def test_action_hash_ignores_the_id_field() -> None:
    with_id = ProposedAction(id="call_abc", tool_name="x", arguments={"a": 1})
    without_id = ProposedAction(tool_name="x", arguments={"a": 1})
    assert compute_action_hash(with_id) == compute_action_hash(without_id)


def test_action_hash_differs_for_different_actions() -> None:
    assert compute_action_hash(ACTION) != compute_action_hash(OTHER_ACTION)


# --- is_affirmative ----------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    ["yes", "Yes!", "yeah, sure", "confirm", "confirmed", "please proceed", "go ahead", "do it"],
)
def test_is_affirmative_detects_common_confirmations(content: str) -> None:
    assert is_affirmative(Message(role="user", content=content)) is True


@pytest.mark.parametrize("content", ["no", "no thanks", "don't do it", "cancel that", "wait"])
def test_is_affirmative_rejects_negative_replies(content: str) -> None:
    assert is_affirmative(Message(role="user", content=content)) is False


def test_is_affirmative_false_for_unrelated_text() -> None:
    assert is_affirmative(Message(role="user", content="what's my order status?")) is False


def test_is_affirmative_false_for_non_user_role() -> None:
    assert is_affirmative(Message(role="assistant", content="yes")) is False


def test_is_affirmative_false_for_empty_content() -> None:
    assert is_affirmative(Message(role="user", content=None)) is False


# --- evaluate_write_gate ------------------------------------------------------


def test_asks_for_confirmation_when_nothing_pending() -> None:
    verdict = evaluate_write_gate(
        ACTION, pending_confirmation=None, last_user_message=None, clause_id="c1", reason="r1"
    )

    assert verdict.confirmed is False
    assert verdict.prompt_message is not None
    assert "cancel_pending_order" in verdict.prompt_message
    assert "c1" in verdict.prompt_message


def test_confirms_when_pending_matches_and_user_said_yes() -> None:
    verdict = evaluate_write_gate(
        ACTION,
        pending_confirmation=ACTION,
        last_user_message=Message(role="user", content="yes"),
        clause_id="c1",
        reason="r1",
    )

    assert verdict.confirmed is True
    assert verdict.prompt_message is None


def test_asks_again_when_pending_action_differs() -> None:
    verdict = evaluate_write_gate(
        ACTION,
        pending_confirmation=OTHER_ACTION,
        last_user_message=Message(role="user", content="yes"),
        clause_id="c1",
        reason="r1",
    )

    assert verdict.confirmed is False


def test_asks_again_when_user_did_not_confirm() -> None:
    verdict = evaluate_write_gate(
        ACTION,
        pending_confirmation=ACTION,
        last_user_message=Message(role="user", content="actually never mind"),
        clause_id="c1",
        reason="r1",
    )

    assert verdict.confirmed is False


# --- DispatchCache -------------------------------------------------------------


def test_dispatch_cache_calls_handler_once_for_repeated_key() -> None:
    cache = DispatchCache()
    calls = []

    def dispatch_fn() -> ToolResult:
        calls.append(1)
        return ToolResult(ok=True, tool_name="cancel_pending_order", result="done")

    result1, cached1 = cache.get_or_dispatch("session-1", 3, ACTION, dispatch_fn)
    result2, cached2 = cache.get_or_dispatch("session-1", 3, ACTION, dispatch_fn)

    assert len(calls) == 1
    assert cached1 is False
    assert cached2 is True
    assert result1 == result2


def test_dispatch_cache_distinguishes_different_turns() -> None:
    cache = DispatchCache()
    calls = []

    def dispatch_fn() -> ToolResult:
        calls.append(1)
        return ToolResult(ok=True, tool_name="cancel_pending_order", result="done")

    cache.get_or_dispatch("session-1", 3, ACTION, dispatch_fn)
    cache.get_or_dispatch("session-1", 4, ACTION, dispatch_fn)

    assert len(calls) == 2


def test_dispatch_cache_distinguishes_different_sessions() -> None:
    cache = DispatchCache()
    calls = []

    def dispatch_fn() -> ToolResult:
        calls.append(1)
        return ToolResult(ok=True, tool_name="cancel_pending_order", result="done")

    cache.get_or_dispatch("session-1", 3, ACTION, dispatch_fn)
    cache.get_or_dispatch("session-2", 3, ACTION, dispatch_fn)

    assert len(calls) == 2
