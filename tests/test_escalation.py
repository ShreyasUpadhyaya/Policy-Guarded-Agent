from __future__ import annotations

from guarded_agent.guardrails.escalation import (
    check_policy_deadlock,
    check_repeated_tool_failure,
    count_consecutive_tool_failures,
)
from guarded_agent.state import Message, ToolCall

FAILED_TOOL = Message(role="tool", content="error!", error=True)
OK_TOOL = Message(role="tool", content="ok", error=False)
PROPOSAL = Message(role="assistant", tool_calls=[ToolCall(name="get_order_details", arguments={})])
TEXT_REPLY = Message(role="assistant", content="Here's what I found.")
USER_MSG = Message(role="user", content="thanks")


def test_empty_conversation_has_no_failures() -> None:
    assert count_consecutive_tool_failures([]) == 0


def test_single_failure_counts_one() -> None:
    assert count_consecutive_tool_failures([FAILED_TOOL]) == 1


def test_trailing_success_resets_to_zero() -> None:
    assert count_consecutive_tool_failures([FAILED_TOOL, FAILED_TOOL, OK_TOOL]) == 0


def test_proposal_messages_do_not_break_the_streak() -> None:
    conversation = [FAILED_TOOL, PROPOSAL, FAILED_TOOL, PROPOSAL, FAILED_TOOL]
    assert count_consecutive_tool_failures(conversation) == 3


def test_text_reply_breaks_the_streak() -> None:
    conversation = [FAILED_TOOL, FAILED_TOOL, TEXT_REPLY, PROPOSAL, FAILED_TOOL]
    assert count_consecutive_tool_failures(conversation) == 1


def test_user_message_breaks_the_streak() -> None:
    conversation = [FAILED_TOOL, FAILED_TOOL, USER_MSG, PROPOSAL, FAILED_TOOL]
    assert count_consecutive_tool_failures(conversation) == 1


def test_check_repeated_tool_failure_below_threshold_does_not_escalate() -> None:
    verdict = check_repeated_tool_failure([FAILED_TOOL, PROPOSAL, FAILED_TOOL], max_consecutive=3)
    assert verdict.should_escalate is False


def test_check_repeated_tool_failure_at_threshold_escalates() -> None:
    conversation = [FAILED_TOOL, PROPOSAL, FAILED_TOOL, PROPOSAL, FAILED_TOOL]
    verdict = check_repeated_tool_failure(conversation, max_consecutive=3)
    assert verdict.should_escalate is True
    assert verdict.reason is not None
    assert "3 consecutive tool failures" in verdict.reason


def test_check_policy_deadlock_below_threshold_does_not_escalate() -> None:
    assert check_policy_deadlock(2, max_consecutive=3).should_escalate is False


def test_check_policy_deadlock_at_threshold_escalates() -> None:
    verdict = check_policy_deadlock(3, max_consecutive=3)
    assert verdict.should_escalate is True
    assert verdict.reason is not None
    assert "3 consecutive policy denials" in verdict.reason
