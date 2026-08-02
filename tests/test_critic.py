from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from guarded_agent.guardrails.critic import (
    _build_user_prompt,
    _extract_json_object,
    _format_clauses,
    _format_history,
    make_llm_critic_check_fn,
)
from guarded_agent.guardrails.policy_retrieval import PolicyContext, RetrievedClause
from guarded_agent.state import Message

CONFIDENT_CONTEXT = PolicyContext(
    clauses=[
        RetrievedClause(
            clause_id="return-delivered-order", text="Refunds go to original payment.", score=0.45
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


def test_format_history_labels_each_role() -> None:
    formatted = _format_history(CONVERSATION)
    assert "user: Where's my order?" in formatted
    assert "tool result: status: shipped" in formatted


def test_format_clauses_lists_confident_clauses_with_ids() -> None:
    formatted = _format_clauses(CONFIDENT_CONTEXT)
    assert "[return-delivered-order]" in formatted


def test_format_clauses_uses_full_text_on_fallback() -> None:
    formatted = _format_clauses(FALLBACK_CONTEXT)
    assert "low confidence" in formatted
    assert "FULL POLICY TEXT HERE" in formatted


def test_build_user_prompt_includes_draft_and_history() -> None:
    prompt = _build_user_prompt(CONVERSATION, "It's on its way!", CONFIDENT_CONTEXT)
    assert "It's on its way!" in prompt
    assert "Where's my order?" in prompt
    assert "return-delivered-order" in prompt


# --- JSON extraction robustness ---------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"approved": true}', '{"approved": true}'),
        ('```json\n{"approved": true}\n```', '{"approved": true}'),
        ('```\n{"approved": true}\n```', '{"approved": true}'),
        ('Sure:\n{"approved": true}\nDone.', '{"approved": true}'),
    ],
)
def test_extract_json_object_handles_common_wrapping(raw: str, expected: str) -> None:
    assert _extract_json_object(raw) == expected


# --- full check_fn round trip, litellm mocked -------------------------------


def test_llm_critic_check_fn_parses_clean_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.critic.litellm.completion",
        lambda **kwargs: _fake_completion_response('{"approved": true, "reason": "Grounded."}'),
    )
    check_fn = make_llm_critic_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, "It's on its way!", CONFIDENT_CONTEXT)

    assert verdict.approved is True
    assert verdict.reason == "Grounded."


def test_llm_critic_check_fn_handles_markdown_fenced_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.critic.litellm.completion",
        lambda **kwargs: _fake_completion_response(
            '```json\n{"approved": false, "reason": "Unsupported claim."}\n```'
        ),
    )
    check_fn = make_llm_critic_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, "Your refund was already sent.", CONFIDENT_CONTEXT)

    assert verdict.approved is False
    assert verdict.reason == "Unsupported claim."


def test_llm_critic_check_fn_fails_closed_on_unparseable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.critic.litellm.completion",
        lambda **kwargs: _fake_completion_response("I'm not sure..."),
    )
    check_fn = make_llm_critic_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, "draft", CONFIDENT_CONTEXT)

    assert verdict.approved is False
    assert "did not parse" in verdict.reason


def test_llm_critic_check_fn_fails_closed_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "guarded_agent.guardrails.critic.litellm.completion",
        lambda **kwargs: _fake_completion_response(None),
    )
    check_fn = make_llm_critic_check_fn("fake-model")

    verdict = check_fn(CONVERSATION, "draft", CONFIDENT_CONTEXT)

    assert verdict.approved is False
    assert "no content" in verdict.reason


def test_llm_critic_check_fn_passes_model_and_temperature_through(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def _capture(**kwargs: Any) -> MagicMock:
        captured.update(kwargs)
        return _fake_completion_response('{"approved": true, "reason": "ok"}')

    monkeypatch.setattr("guarded_agent.guardrails.critic.litellm.completion", _capture)
    check_fn = make_llm_critic_check_fn("claude-haiku-4-5-20251001", temperature=0.0)

    check_fn(CONVERSATION, "draft", CONFIDENT_CONTEXT)

    assert captured["model"] == "claude-haiku-4-5-20251001"
    assert captured["temperature"] == 0.0
    assert captured["messages"][0]["role"] == "system"
