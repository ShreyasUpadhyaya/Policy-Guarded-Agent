from __future__ import annotations

import json
import re
from collections.abc import Callable

import litellm
from pydantic import BaseModel, ValidationError

from guarded_agent.guardrails.policy_retrieval import PolicyContext
from guarded_agent.state import Message

CRITIC_SYSTEM_PROMPT = """
You are a critic reviewing a customer service agent's drafted reply before it is sent
to the customer. Check for two things:
1. Unsupported claims: does the draft assert a fact (an order status, a refund amount,
   a policy detail) that isn't grounded in the conversation history or tool results
   shown to you?
2. Policy drift: does the draft contradict or ignore the policy clauses given?

Respond with ONLY a JSON object, no other text, no markdown code fences, with exactly
these keys: "approved" (boolean), "reason" (one sentence explaining the verdict either
way).
""".strip()

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class CriticVerdict(BaseModel):
    approved: bool
    reason: str


def _extract_json_object(text: str) -> str:
    """Same defensive extraction as guardrails/policy_checker.py -- kept as a
    separate small copy rather than a shared import so each guardrail
    module stays independently readable; the logic is a few lines, not
    worth a cross-module dependency for."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    match = _JSON_OBJECT_PATTERN.search(stripped)
    return match.group(0) if match else stripped


def _fail_closed(reason: str) -> CriticVerdict:
    """A critic check that couldn't produce a valid verdict fails to
    not-approved, never to a silent approval -- same fail-safe-forcing
    rationale as guardrails/policy_checker.py's _fail_closed."""
    return CriticVerdict(approved=False, reason=reason)


def _format_history(conversation: list[Message]) -> str:
    lines = []
    for message in conversation:
        if message.role == "user":
            lines.append(f"user: {message.content}")
        elif message.role == "tool":
            lines.append(f"tool result: {message.content}")
        elif message.content:
            lines.append(f"assistant: {message.content}")
    return "\n".join(lines)


def _format_clauses(context: PolicyContext) -> str:
    if context.used_fallback:
        return (
            f"[Full policy text -- retrieval fallback: {context.fallback_reason}]\n"
            f"{context.full_policy_text}"
        )
    return "\n\n".join(f"[{clause.clause_id}] {clause.text}" for clause in context.clauses)


def _build_user_prompt(conversation: list[Message], draft: str, context: PolicyContext) -> str:
    return (
        "Conversation so far:\n"
        f"{_format_history(conversation)}\n\n"
        "Drafted reply to review:\n"
        f"{draft}\n\n"
        "Relevant policy clauses:\n"
        f"{_format_clauses(context)}"
    )


CriticCheckFn = Callable[[list[Message], str, PolicyContext], CriticVerdict]
"""(conversation so far, drafted response, retrieved policy context) -> CriticVerdict.

Injectable so tests can supply a fixed stub instead of a real LLM call,
matching graph.py's GenerateFn and policy_checker.py's PolicyCheckFn.
"""


def make_llm_critic_check_fn(model: str, temperature: float = 0.0) -> CriticCheckFn:
    """Real critic checking via a direct litellm call, independent of tau2
    for the same reason as guardrails/policy_checker.py's
    make_llm_policy_check_fn: this evaluates an already-drafted response
    rather than proposing a tool call, so it needs none of tau2's
    tool-calling message format."""

    def _check(conversation: list[Message], draft: str, context: PolicyContext) -> CriticVerdict:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(conversation, draft, context)},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return _fail_closed("critic LLM call returned no content")

        json_text = _extract_json_object(content)
        try:
            return CriticVerdict.model_validate_json(json_text)
        except (ValidationError, json.JSONDecodeError) as exc:
            return _fail_closed(f"critic response did not parse as a valid verdict: {exc}")

    return _check


def check_response(
    conversation: list[Message], draft: str, context: PolicyContext, check_fn: CriticCheckFn
) -> CriticVerdict:
    """Thin entry point, mirrors guardrails/policy_checker.py's check_policy."""
    return check_fn(conversation, draft, context)
