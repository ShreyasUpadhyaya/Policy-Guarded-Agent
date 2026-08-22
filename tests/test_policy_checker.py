from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from guarded_agent.guardrails.policy_checker import (
    _build_user_prompt,
    _extract_json_object,
    _format_clauses,
    _response_cost,
    make_llm_policy_check_fn,
)
from guarded_agent.guardrails.policy_retrieval import PolicyContext, RetrievedClause
from guarded_agent.state import Message, ProposedAction

CONFIDENT_CONTEXT = PolicyContext(
    clauses=[
        RetrievedClause(
            clause_id="return-delivered-order", text="Refunds go to original payment.", score=0.45
        ),
        RetrievedClause(
            clause_id="cancel-pending-order",
            text="Cancellations get an immediate refund.",
            score=0.30,
        ),
    ],
    used_fallback=False,
)

FALLBACK_CONTEXT = PolicyContext(
    clauses=[],
    used_fallback=True,
    fallback_reason="low confidence",
    full_policy_text="FULL POLICY TEXT HERE",
)

ACTION = ProposedAction(
    id="call_1", tool_name="issue_refund", arguments={"order_id": "123", "amount": 42.0}
)

CONVERSATION = [
    Message(role="user", content="Where's my order?"),
    Message(role="tool", content="status: shipped"),
]


def _fake_completion_response(content: str | None) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


# --- pure formatting/prompt-building logic ---------------------------------


def test_format_clauses_lists_confident_clauses_with_ids() -> None:
    formatted = _format_clauses(CONFIDENT_CONTEXT)
    assert "[return-delivered-order]" in formatted
    assert "[cancel-pending-order]" in formatted
    assert "Refunds go to original payment." in formatted


def test_format_clauses_uses_full_text_on_fallback() -> None:
    formatted = _format_clauses(FALLBACK_CONTEXT)
    assert "low confidence" in formatted
    assert "FULL POLICY TEXT HERE" in formatted


def test_build_user_prompt_includes_action_name_and_arguments() -> None:
    prompt = _build_user_prompt(CONVERSATION, ACTION, CONFIDENT_CONTEXT)
    assert "issue_refund" in prompt
    assert "123" in prompt
    assert "return-delivered-order" in prompt


def test_build_user_prompt_includes_conversation_history() -> None:
    prompt = _build_user_prompt(CONVERSATION, ACTION, CONFIDENT_CONTEXT)
    assert "Where's my order?" in prompt
    assert "status: shipped" in prompt


# --- JSON extraction robustness ---------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"verdict": "ALLOW"}', '{"verdict": "ALLOW"}'),
        ('```json\n{"verdict": "ALLOW"}\n```', '{"verdict": "ALLOW"}'),
        ('```\n{"verdict": "ALLOW"}\n```', '{"verdict": "ALLOW"}'),
        (
            'Sure, here is my answer:\n{"verdict": "ALLOW"}\nLet me know if needed.',
            '{"verdict": "ALLOW"}',
        ),
        ('  {"verdict": "ALLOW"}  ', '{"verdict": "ALLOW"}'),
    ],
)
def test_extract_json_object_handles_common_wrapping(raw: str, expected: str) -> None:
    assert _extract_json_object(raw) == expected


# --- full check_fn round trip, litellm mocked -------------------------------


def test_llm_policy_check_fn_parses_clean_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion",
        lambda **kwargs: _fake_completion_response(
            '{"verdict": "ALLOW", "clause_id": "return-delivered-order", "reason": "OK."}'
        ),
    )
    check_fn = make_llm_policy_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert verdict.verdict == "ALLOW"
    assert verdict.clause_id == "return-delivered-order"
    assert verdict.reason == "OK."


def test_llm_policy_check_fn_handles_markdown_fenced_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion",
        lambda **kwargs: _fake_completion_response(
            '```json\n{"verdict": "DENY", "clause_id": "cancel-pending-order", "reason": "No"}\n```'
        ),
    )
    check_fn = make_llm_policy_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert verdict.verdict == "DENY"


def test_llm_policy_check_fn_fails_closed_on_unparseable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion",
        lambda **kwargs: _fake_completion_response("I'm not sure, let me think about it..."),
    )
    check_fn = make_llm_policy_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert verdict.verdict == "DENY"
    assert "did not parse" in verdict.reason


def test_llm_policy_check_fn_fails_closed_on_invalid_verdict_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion",
        lambda **kwargs: _fake_completion_response(
            '{"verdict": "MAYBE", "clause_id": "x", "reason": "unsure"}'
        ),
    )
    check_fn = make_llm_policy_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert verdict.verdict == "DENY"


def test_llm_policy_check_fn_fails_closed_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion",
        lambda **kwargs: _fake_completion_response(None),
    )
    check_fn = make_llm_policy_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert verdict.verdict == "DENY"
    assert "no content" in verdict.reason


def test_llm_policy_check_fn_passes_model_and_temperature_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return _fake_completion_response('{"verdict": "ALLOW", "clause_id": "x", "reason": "y"}')

    monkeypatch.setattr("guarded_agent.guardrails.policy_checker.litellm.completion", _capture)
    check_fn = make_llm_policy_check_fn("claude-haiku-4-5-20251001", temperature=0.0)

    check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["temperature"] == 0.0
    assert captured["messages"][0]["role"] == "system"


# --- on_cost: this call is real, separately-billed spend invisible to tau2
# (it never goes through tau2's generate()) -- verified live against a real
# Haiku ablation run that this cost was being silently dropped entirely
# (PLAN.md commit 24 calibration notes). ----------------------------------


def test_llm_policy_check_fn_calls_on_cost_with_real_cost_and_usage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = _fake_completion_response('{"verdict": "ALLOW", "clause_id": "x", "reason": "y"}')
    response.usage.prompt_tokens = 120
    response.usage.completion_tokens = 30
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion", lambda **kwargs: response
    )
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion_cost",
        lambda **kwargs: 0.0042,
    )
    calls: list[tuple[float, dict[str, int] | None]] = []
    check_fn = make_llm_policy_check_fn("fake-model", on_cost=lambda c, u: calls.append((c, u)))

    check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert calls == [(0.0042, {"prompt_tokens": 120, "completion_tokens": 30})]


def test_llm_policy_check_fn_calls_on_cost_even_when_response_fails_to_parse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The LLM call already happened and cost money regardless of whether
    the response was usable -- a fail-closed DENY must not also silently
    drop the cost of the call that produced it."""
    response = _fake_completion_response("not valid json")
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion", lambda **kwargs: response
    )
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion_cost", lambda **kwargs: 0.001
    )
    calls: list[float] = []
    check_fn = make_llm_policy_check_fn("fake-model", on_cost=lambda c, u: calls.append(c))

    verdict = check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)

    assert verdict.verdict == "DENY"
    assert calls == [0.001]


def test_llm_policy_check_fn_without_on_cost_never_calls_completion_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """on_cost is optional -- when the caller doesn't ask for cost tracking,
    completion_cost must not even be invoked, not just have its result
    discarded (it can raise for models with no pricing data)."""
    response = _fake_completion_response('{"verdict": "ALLOW", "clause_id": "x", "reason": "y"}')
    monkeypatch.setattr(
        "guarded_agent.guardrails.policy_checker.litellm.completion", lambda **kwargs: response
    )

    def _boom(**kwargs: Any) -> float:
        raise AssertionError("completion_cost must not be called without on_cost")

    monkeypatch.setattr("guarded_agent.guardrails.policy_checker.litellm.completion_cost", _boom)
    check_fn = make_llm_policy_check_fn("fake-model")

    check_fn(CONVERSATION, ACTION, CONFIDENT_CONTEXT)  # must not raise


def test_response_cost_defaults_to_zero_when_completion_cost_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _raise(**kwargs: Any) -> float:
        raise ValueError("no pricing data for this model")

    monkeypatch.setattr("guarded_agent.guardrails.policy_checker.litellm.completion_cost", _raise)

    assert _response_cost(MagicMock()) == 0.0
