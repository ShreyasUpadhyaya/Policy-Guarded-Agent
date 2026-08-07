from __future__ import annotations

import json
import re
from collections.abc import Callable

import litellm
from pydantic import ValidationError

from guarded_agent.guardrails.policy_retrieval import PolicyContext
from guarded_agent.state import Message, PolicyVerdict, ProposedAction

POLICY_CHECK_SYSTEM_PROMPT = """
You are a policy compliance checker for a customer service agent. You will be given
the conversation so far, a proposed tool call, and the relevant policy clauses. Decide
whether the action should be ALLOWED, DENIED, or requires the user's explicit
confirmation first.

Use the conversation so far to check whether any prerequisite the policy clauses
mention (e.g. user authentication) was already satisfied earlier in this same
conversation -- for example, an earlier successful call that establishes identity
means an authentication clause is already satisfied for later actions in the same
session, not a reason to deny them again.

Rules:
- ALLOW: the action is clearly permitted by the policy clauses given (accounting for
  what has already happened in the conversation), and no further confirmation is
  required by those clauses.
- DENY: the action violates the policy clauses given, or is not covered by them.
- NEEDS_CONFIRMATION: the action would be permitted, but the policy requires explicit
  user confirmation before it proceeds, and nothing in the conversation indicates
  that confirmation was already obtained.

You must cite the id of exactly one policy clause that most directly supports your
verdict. If the provided clauses don't clearly cover the action, deny it -- never
allow an action you cannot ground in a specific clause id.

Respond with ONLY a JSON object, no other text, no markdown code fences, with exactly
these keys: "verdict" (one of "ALLOW", "DENY", "NEEDS_CONFIRMATION"), "clause_id"
(string, the id of the clause you cited), "reason" (one sentence).
""".strip()

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def _format_clauses(context: PolicyContext) -> str:
    if context.used_fallback:
        return (
            f"[Full policy text -- retrieval fallback: {context.fallback_reason}]\n"
            f"{context.full_policy_text}"
        )
    return "\n\n".join(f"[{clause.clause_id}] {clause.text}" for clause in context.clauses)


def _format_history(conversation: list[Message]) -> str:
    """Same small formatting helper as guardrails/critic.py's -- kept as a
    separate copy rather than a shared import so each guardrail module stays
    independently readable; the logic is a few lines, not worth a
    cross-module dependency for."""
    lines = []
    for message in conversation:
        if message.role == "user":
            lines.append(f"user: {message.content}")
        elif message.role == "tool":
            lines.append(f"tool result: {message.content}")
        elif message.content:
            lines.append(f"assistant: {message.content}")
    return "\n".join(lines)


def _build_user_prompt(
    conversation: list[Message], action: ProposedAction, context: PolicyContext
) -> str:
    return (
        "Conversation so far:\n"
        f"{_format_history(conversation)}\n\n"
        "Proposed tool call:\n"
        f"name: {action.tool_name}\n"
        f"arguments: {json.dumps(action.arguments)}\n\n"
        "Relevant policy clauses:\n"
        f"{_format_clauses(context)}"
    )


def _extract_json_object(text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response.

    Models are asked to respond with only JSON, but reliably strip markdown
    code fences and any leading/trailing prose anyway rather than assume
    compliance -- this is a known, common failure mode (tau2's own
    llm_utils has an equivalent extractor for the same reason).
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    match = _JSON_OBJECT_PATTERN.search(stripped)
    return match.group(0) if match else stripped


def _fail_closed(reason: str) -> PolicyVerdict:
    """A policy check that couldn't produce a valid verdict fails to DENY,
    never to ALLOW -- fail-safe forcing, per CLAUDE.md: a degraded input must
    never resolve to a silent ALLOW. No retry loop: one bad response is
    enough to deny and move on, not enough to justify an unbounded retry.
    """
    return PolicyVerdict(verdict="DENY", clause_id="", reason=reason)


PolicyCheckFn = Callable[[list[Message], ProposedAction, PolicyContext], PolicyVerdict]
"""(conversation so far, proposed action, retrieved policy context) -> PolicyVerdict.

Injectable so tests can supply a fixed stub or a fixture-recorded response
instead of a real LLM call, matching graph.py's GenerateFn pattern.

Takes the conversation for the same reason guardrails/critic.py's
CriticCheckFn does (and always has): a policy clause like "authenticate the
user first" is only checkable against what's already happened in *this*
session -- verified live (PLAN.md commit 21 v2 smoke run) that without it,
the checker has no way to know authentication already succeeded earlier in
the same conversation and denies every subsequent action that clause
covers, over and over, for the rest of the session.
"""


def make_llm_policy_check_fn(model: str, temperature: float = 0.0) -> PolicyCheckFn:
    """Real policy checking via a direct litellm call.

    Deliberately not routed through tau2.utils.llm_utils.generate() like
    graph.py's agent node: this doesn't need tau2's tool-calling message
    format at all (it's evaluating an already-proposed action, not
    proposing one), so keeping it on plain litellm keeps this guardrail
    module independent of tau2, consistent with the rest of guardrails/.
    """

    def _check(
        conversation: list[Message], action: ProposedAction, context: PolicyContext
    ) -> PolicyVerdict:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": POLICY_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(conversation, action, context)},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return _fail_closed("policy check LLM call returned no content")

        json_text = _extract_json_object(content)
        try:
            return PolicyVerdict.model_validate_json(json_text)
        except (ValidationError, json.JSONDecodeError) as exc:
            return _fail_closed(f"policy check response did not parse as a valid verdict: {exc}")

    return _check


def check_policy(
    conversation: list[Message],
    action: ProposedAction,
    context: PolicyContext,
    check_fn: PolicyCheckFn,
) -> PolicyVerdict:
    """Thin entry point so callers don't need to know check_fn exists as a
    concept -- mirrors ToolRegistry.dispatch's shape (validated inputs in,
    structured result out)."""
    return check_fn(conversation, action, context)
