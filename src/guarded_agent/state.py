from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, Any]


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class ProposedAction(BaseModel):
    """A tool call the agent wants to make, pending policy check / write gate."""

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

    def increment(self, *, steps: int = 0, tool_calls: int = 0, tokens: int = 0) -> BudgetCounters:
        """Return a new BudgetCounters with the given deltas applied."""
        return self.model_copy(
            update={
                "steps_used": self.steps_used + steps,
                "tool_calls_used": self.tool_calls_used + tool_calls,
                "tokens_used": self.tokens_used + tokens,
            }
        )


class AgentState(BaseModel):
    conversation: list[Message] = Field(default_factory=list)
    plan: str | None = None
    proposed_action: ProposedAction | None = None
    policy_verdict: PolicyVerdict | None = None
    budget: BudgetCounters = Field(default_factory=BudgetCounters)
    escalated: bool = False
    escalation_reason: str | None = None

    def add_message(self, message: Message) -> AgentState:
        """Return a new AgentState with `message` appended to the conversation."""
        return self.model_copy(update={"conversation": [*self.conversation, message]})

    def escalate(self, reason: str) -> AgentState:
        """Return a new AgentState marked as escalated, with the given reason."""
        return self.model_copy(update={"escalated": True, "escalation_reason": reason})
