from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable

from pydantic import BaseModel

from guarded_agent.state import Message, ProposedAction
from guarded_agent.tools.registry import ToolResult

_AFFIRMATIVE_PATTERN = re.compile(
    r"\b(yes|yep|yeah|confirm(?:ed)?|proceed|go ahead|correct|do it)\b", re.IGNORECASE
)
_NEGATIVE_PATTERN = re.compile(r"\b(no|nope|don'?t|cancel|stop|wait)\b", re.IGNORECASE)


class WriteGateVerdict(BaseModel):
    confirmed: bool
    prompt_message: str | None = None
    """Set when confirmed=False: the message to show the user, presenting
    the action details and asking for explicit confirmation."""


def compute_action_hash(action: ProposedAction) -> str:
    """Deterministic hash of a proposed action's tool name + arguments.

    Used both to key idempotent dispatch and to check whether a
    newly-proposed action is the same one a pending confirmation was asked
    about -- content-based, not id-based, since the LLM assigns a fresh
    tool_call id on each turn even when re-proposing the same call.
    """
    payload = json.dumps(
        {"tool_name": action.tool_name, "arguments": action.arguments}, sort_keys=True
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def is_affirmative(message: Message) -> bool:
    """Best-effort detection of an explicit "yes" reply.

    The retail policy's own wording is "obtain explicit user confirmation
    (yes) to proceed" -- checking for an explicit affirmative is policy-
    grounded, not an arbitrary heuristic. A real deployment might want an
    LLM-based confirmation classifier for ambiguous replies; this keeps the
    gate itself pure and zero-I/O (CLAUDE.md's guardrail purity rule),
    trading precision on ambiguous phrasing for that.
    """
    if message.role != "user" or not message.content:
        return False
    text = message.content.strip()
    if _NEGATIVE_PATTERN.search(text):
        return False
    return bool(_AFFIRMATIVE_PATTERN.search(text))


def evaluate_write_gate(
    action: ProposedAction,
    pending_confirmation: ProposedAction | None,
    last_user_message: Message | None,
    clause_id: str,
    reason: str,
) -> WriteGateVerdict:
    """Decide whether a mutating action can proceed now, or must wait for
    explicit confirmation first.

    Zero I/O: takes everything it needs as plain data, per CLAUDE.md's
    guardrail purity rule -- no clock, no LLM call, no network.
    """
    same_action_pending = pending_confirmation is not None and compute_action_hash(
        pending_confirmation
    ) == compute_action_hash(action)
    if same_action_pending and last_user_message is not None and is_affirmative(last_user_message):
        return WriteGateVerdict(confirmed=True)

    prompt = (
        f"Before I proceed: I'd like to call `{action.tool_name}` with "
        f"{json.dumps(action.arguments)}. This is based on policy [{clause_id}]: {reason}. "
        "Shall I proceed? (yes/no)"
    )
    return WriteGateVerdict(confirmed=False, prompt_message=prompt)


class DispatchCache:
    """In-memory idempotency cache for mutating dispatches.

    Keyed by (session_id, turn, action_hash) per PLAN.md commit 17: a
    retried dispatch for the same key returns the cached result instead of
    invoking the handler twice. Scoped to one instance per conversation
    (one per build_graph() call), not shared across conversations.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[str, int, str], ToolResult] = {}

    def get_or_dispatch(
        self,
        session_id: str,
        turn: int,
        action: ProposedAction,
        dispatch_fn: Callable[[], ToolResult],
    ) -> tuple[ToolResult, bool]:
        """Returns (result, was_cached)."""
        key = (session_id, turn, compute_action_hash(action))
        if key in self._cache:
            return self._cache[key], True
        result = dispatch_fn()
        self._cache[key] = result
        return result, False
