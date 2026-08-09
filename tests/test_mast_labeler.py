from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from analysis.mast_labeler import (
    JudgeVerdict,
    _extract_json_object,
    check_budget_breach,
    check_missing_confirmation,
    check_non_termination,
    check_schema_violation,
    label_failed_traces,
    label_trace,
    make_llm_judge_fn,
    run_deterministic_checks,
)
from analysis.trace_loader import ActionCheck, Step, ToolCall, Trace

BASE_TRACE_KWARGS: dict[str, Any] = {
    "id": "sim-1",
    "task_id": "t1",
    "trial": 0,
    "domain": "retail",
    "agent_cost": 0.01,
    "user_cost": 0.005,
    "reward": 0.0,
    "reward_breakdown": {"DB": 0.0, "NL_ASSERTION": 0.0},
}


def _trace(steps: list[Step], termination_reason: str = "user_stop", **overrides: Any) -> Trace:
    kwargs = {**BASE_TRACE_KWARGS, "termination_reason": termination_reason, "steps": steps}
    kwargs.update(overrides)
    return Trace(**kwargs)


def _step(role: str, content: str | None = None, tool_calls: list[ToolCall] | None = None) -> Step:
    return Step(role=role, content=content, tool_calls=tool_calls, turn_idx=0)


# --- deterministic checks ----------------------------------------------------


def test_schema_violation_detected_via_registry_error_wording() -> None:
    trace = _trace([_step("assistant", "Arguments for 'issue_refund' failed schema validation.")])
    assert check_schema_violation(trace) is True


def test_schema_violation_false_for_an_ordinary_reply() -> None:
    trace = _trace([_step("assistant", "Happy to help with that!")])
    assert check_schema_violation(trace) is False


@pytest.mark.parametrize(
    "reason", ["max_steps", "timeout", "too_many_errors", "context_window_exceeded"]
)
def test_budget_breach_true_for_resource_cap_terminations(reason: str) -> None:
    trace = _trace([_step("assistant", "hi")], termination_reason=reason)
    assert check_budget_breach(trace) is True


@pytest.mark.parametrize("reason", ["user_stop", "agent_stop"])
def test_budget_breach_false_for_natural_terminations(reason: str) -> None:
    trace = _trace([_step("assistant", "hi")], termination_reason=reason)
    assert check_budget_breach(trace) is False


@pytest.mark.parametrize("reason", ["user_stop", "agent_stop"])
def test_non_termination_false_for_natural_stops(reason: str) -> None:
    trace = _trace([_step("assistant", "hi")], termination_reason=reason)
    assert check_non_termination(trace) is False


@pytest.mark.parametrize("reason", ["max_steps", "agent_error", "infrastructure_error"])
def test_non_termination_true_for_anything_else(reason: str) -> None:
    trace = _trace([_step("assistant", "hi")], termination_reason=reason)
    assert check_non_termination(trace) is True


WRITE_CALL = [ToolCall(id="c1", name="issue_refund", arguments={}, requestor="assistant")]
WRITE_ACTION_CHECK = ActionCheck(name="issue_refund", tool_type="write", action_match=True)
UNMATCHED_WRITE_ACTION_CHECK = ActionCheck(
    name="issue_refund", tool_type="write", action_match=False
)
READ_ACTION_CHECK = ActionCheck(name="get_order_details", tool_type="read", action_match=True)


def test_missing_confirmation_true_when_no_user_message_precedes_a_write() -> None:
    trace = _trace(
        [_step("assistant", tool_calls=WRITE_CALL)],
        action_checks=[WRITE_ACTION_CHECK],
    )
    assert check_missing_confirmation(trace) is True


def test_missing_confirmation_true_when_prior_user_message_is_not_affirmative() -> None:
    trace = _trace(
        [
            _step("user", "I'd like a refund please."),
            _step("assistant", tool_calls=WRITE_CALL),
        ],
        action_checks=[WRITE_ACTION_CHECK],
    )
    assert check_missing_confirmation(trace) is True


def test_missing_confirmation_false_when_user_explicitly_confirmed() -> None:
    trace = _trace(
        [
            _step("assistant", "Shall I proceed? (yes/no)"),
            _step("user", "Yes, go ahead."),
            _step("assistant", tool_calls=WRITE_CALL),
        ],
        action_checks=[WRITE_ACTION_CHECK],
    )
    assert check_missing_confirmation(trace) is False


def test_missing_confirmation_false_when_no_write_actions_matched() -> None:
    trace = _trace(
        [_step("assistant", tool_calls=WRITE_CALL)],
        action_checks=[UNMATCHED_WRITE_ACTION_CHECK],
    )
    assert check_missing_confirmation(trace) is False


def test_missing_confirmation_false_for_read_only_traces() -> None:
    read_call = [ToolCall(id="c1", name="get_order_details", arguments={}, requestor="assistant")]
    trace = _trace([_step("assistant", tool_calls=read_call)], action_checks=[READ_ACTION_CHECK])
    assert check_missing_confirmation(trace) is False


def test_run_deterministic_checks_returns_every_label_that_fires() -> None:
    trace = _trace(
        [_step("assistant", tool_calls=WRITE_CALL)],
        termination_reason="max_steps",
        action_checks=[WRITE_ACTION_CHECK],
    )
    labels = run_deterministic_checks(trace)
    assert set(labels) == {"budget_breach", "non_termination", "missing_confirmation"}


def test_run_deterministic_checks_empty_when_nothing_fires() -> None:
    trace = _trace(
        [
            _step("assistant", "Shall I proceed? (yes/no)"),
            _step("user", "yes"),
            _step("assistant", tool_calls=WRITE_CALL),
        ],
        action_checks=[WRITE_ACTION_CHECK],
    )
    assert run_deterministic_checks(trace) == []


# --- label_trace / label_failed_traces ---------------------------------------


def test_label_trace_skips_judge_when_deterministic_checks_fire() -> None:
    trace = _trace([_step("assistant", "hi")], termination_reason="max_steps")

    def _never_judge(t: Trace) -> JudgeVerdict:
        raise AssertionError("judge_fn should never be called when deterministic checks fire")

    label = label_trace(trace, judge_fn=_never_judge)

    assert label.deterministic_labels == ["budget_breach", "non_termination"]
    assert label.judge_label is None


def test_label_trace_calls_judge_only_for_the_residual() -> None:
    trace = _trace([_step("assistant", "hi")], termination_reason="user_stop")

    def _stub_judge(t: Trace) -> JudgeVerdict:
        return JudgeVerdict(label="FM-1.1", reason="disobeyed the task spec")

    label = label_trace(trace, judge_fn=_stub_judge)

    assert label.deterministic_labels == []
    assert label.judge_label == "FM-1.1"
    assert label.judge_reason == "disobeyed the task spec"


def test_label_failed_traces_skips_successful_traces() -> None:
    failed = _trace([_step("assistant", "hi")], reward=0.0)
    succeeded = _trace([_step("assistant", "hi")], reward=1.0, id="sim-2")

    labels = label_failed_traces([failed, succeeded])

    assert len(labels) == 1
    assert labels[0].trace_id == "sim-1"


# --- JSON extraction / LLM judge round trip ----------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('{"label": "FM-1.1"}', '{"label": "FM-1.1"}'),
        ('```json\n{"label": "FM-1.1"}\n```', '{"label": "FM-1.1"}'),
        ('Sure:\n{"label": "FM-1.1"}\nDone.', '{"label": "FM-1.1"}'),
    ],
)
def test_extract_json_object_handles_common_wrapping(raw: str, expected: str) -> None:
    assert _extract_json_object(raw) == expected


def _fake_completion_response(content: str | None) -> MagicMock:
    message = MagicMock()
    message.content = content
    choice = MagicMock()
    choice.message = message
    response = MagicMock()
    response.choices = [choice]
    return response


def test_llm_judge_fn_parses_clean_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "analysis.mast_labeler.litellm.completion",
        lambda **kwargs: _fake_completion_response(
            '{"label": "FM-3.1", "reason": "Handed off before finishing."}'
        ),
    )
    judge_fn = make_llm_judge_fn("fake-model")

    verdict = judge_fn(_trace([_step("assistant", "hi")]))

    assert verdict.label == "FM-3.1"
    assert verdict.reason == "Handed off before finishing."


def test_llm_judge_fn_fails_to_unknown_on_unparseable_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "analysis.mast_labeler.litellm.completion",
        lambda **kwargs: _fake_completion_response("not json at all"),
    )
    judge_fn = make_llm_judge_fn("fake-model")

    verdict = judge_fn(_trace([_step("assistant", "hi")]))

    assert verdict.label == "unknown"


def test_llm_judge_fn_fails_to_unknown_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "analysis.mast_labeler.litellm.completion",
        lambda **kwargs: _fake_completion_response(None),
    )
    judge_fn = make_llm_judge_fn("fake-model")

    verdict = judge_fn(_trace([_step("assistant", "hi")]))

    assert verdict.label == "unknown"
    assert "no content" in verdict.reason
