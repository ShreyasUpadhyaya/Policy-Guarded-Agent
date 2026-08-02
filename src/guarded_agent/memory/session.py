from __future__ import annotations

from pydantic import BaseModel

from guarded_agent.memory.retention import redact_pii
from guarded_agent.state import AgentState


class CaseRecord(BaseModel):
    """A redacted summary of one finished session (resolved or escalated),
    suitable for the case store (PLAN.md commit 19).

    Deliberately narrow: never carries the raw conversation, only
    PII-redacted issue/resolution summaries and structural facts. See
    docs/MEMORY.md for the full retention policy this shape enforces.
    """

    session_id: str
    issue_summary: str | None = None
    resolution_summary: str | None = None
    resolved: bool
    escalation_reason: str | None = None
    tool_names_used: list[str] = []


def _first_user_message(state: AgentState) -> str | None:
    for message in state.conversation:
        if message.role == "user":
            return message.content
    return None


def _last_assistant_text(state: AgentState) -> str | None:
    for message in reversed(state.conversation):
        if message.role == "assistant" and message.content:
            return message.content
    return None


def _distinct_tool_names(state: AgentState) -> list[str]:
    names: list[str] = []
    for message in state.conversation:
        if message.role == "assistant" and message.tool_calls:
            for call in message.tool_calls:
                if call.name not in names:
                    names.append(call.name)
    return names


def build_case_record(state: AgentState, session_id: str) -> CaseRecord:
    """Derive a storable case summary from a session's final state.

    Pure function of state -- no clock, no I/O. The only two free-text
    fields (issue/resolution) are passed through redact_pii before they're
    assigned, so nothing built here can carry raw PII into the case store.
    """
    return CaseRecord(
        session_id=session_id,
        issue_summary=redact_pii(_first_user_message(state)),
        resolution_summary=redact_pii(_last_assistant_text(state)),
        resolved=not state.escalated,
        escalation_reason=state.escalation_reason,
        tool_names_used=_distinct_tool_names(state),
    )
