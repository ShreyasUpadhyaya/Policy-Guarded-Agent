from __future__ import annotations

from pydantic import BaseModel

from guarded_agent.state import Message


class EscalationVerdict(BaseModel):
    should_escalate: bool
    reason: str | None = None


def count_consecutive_tool_failures(conversation: list[Message]) -> int:
    """Count trailing consecutive failed tool results at the end of the
    conversation.

    Derived fresh from history each time, not accumulated as separate
    mutable state, so it can never drift out of sync with the messages
    that justify it -- a pure function of state, per CLAUDE.md's guardrail
    purity rule. A message proposing a tool call (assistant, tool_calls
    set, no text) doesn't break the streak -- it's the "asking" half of a
    result that's already counted or about to be; only a successful tool
    result, a text reply, or a user message ends it.
    """
    count = 0
    for message in reversed(conversation):
        if message.role == "tool":
            if message.error:
                count += 1
                continue
            break
        if message.role == "assistant" and message.tool_calls is not None and not message.content:
            continue
        break
    return count


def check_repeated_tool_failure(
    conversation: list[Message], max_consecutive: int
) -> EscalationVerdict:
    count = count_consecutive_tool_failures(conversation)
    if count >= max_consecutive:
        return EscalationVerdict(should_escalate=True, reason=f"{count} consecutive tool failures")
    return EscalationVerdict(should_escalate=False)


def check_policy_deadlock(consecutive_denials: int, max_consecutive: int) -> EscalationVerdict:
    if consecutive_denials >= max_consecutive:
        return EscalationVerdict(
            should_escalate=True, reason=f"{consecutive_denials} consecutive policy denials"
        )
    return EscalationVerdict(should_escalate=False)
