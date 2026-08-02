from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    id: str = ""
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    """For role="tool": which proposed ToolCall.id this message answers."""
    error: bool = False
    """For role="tool": whether the tool call this message reports on failed."""


class ProposedAction(BaseModel):
    """A tool call the agent wants to make, pending policy check / write gate."""

    id: str = ""
    tool_name: str
    arguments: dict[str, Any]


class PolicyVerdict(BaseModel):
    verdict: Literal["ALLOW", "DENY", "NEEDS_CONFIRMATION"]
    clause_id: str
    reason: str


class BudgetCounters(BaseModel):
    steps_used: int = 0
    tool_calls_used: int = 0
    tokens_used: int = 0
    elapsed_seconds: float = 0.0

    def increment(self, *, steps: int = 0, tool_calls: int = 0, tokens: int = 0) -> BudgetCounters:
        """Return a new BudgetCounters with the given deltas applied."""
        return self.model_copy(
            update={
                "steps_used": self.steps_used + steps,
                "tool_calls_used": self.tool_calls_used + tool_calls,
                "tokens_used": self.tokens_used + tokens,
            }
        )

    def with_elapsed(self, seconds: float) -> BudgetCounters:
        """Return a new BudgetCounters with elapsed_seconds set to `seconds`.

        Set, not incremented: elapsed time is a running total since
        conversation start, computed fresh each turn by whoever tracks the
        start time (the adapter) -- not something this pure model can
        compute itself without reading a clock.
        """
        return self.model_copy(update={"elapsed_seconds": seconds})


class AgentState(BaseModel):
    conversation: list[Message] = Field(default_factory=list)
    plan: str | None = None
    proposed_action: ProposedAction | None = None
    policy_verdict: PolicyVerdict | None = None
    pending_confirmation: ProposedAction | None = None
    """A mutating action the write gate already presented for confirmation,
    awaiting the user's explicit yes/no before it can be proposed again and
    let through."""
    consecutive_policy_denials: int = 0
    """Incremented by the policy gate on DENY, reset on any other verdict.
    Tool-failure streaks aren't tracked this way -- they're derived fresh
    from conversation history (see guardrails/escalation.py) since
    Message.error already carries everything needed, without a second
    mutable counter that could drift out of sync with the messages
    themselves."""
    budget: BudgetCounters = Field(default_factory=BudgetCounters)
    escalated: bool = False
    escalation_reason: str | None = None

    def add_message(self, message: Message) -> AgentState:
        """Return a new AgentState with `message` appended to the conversation."""
        return self.model_copy(update={"conversation": [*self.conversation, message]})

    def escalate(self, reason: str) -> AgentState:
        """Return a new AgentState marked as escalated, with the given reason."""
        return self.model_copy(update={"escalated": True, "escalation_reason": reason})
