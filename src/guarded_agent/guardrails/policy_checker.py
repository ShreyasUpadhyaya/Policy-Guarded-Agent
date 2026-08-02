from __future__ import annotations

import json
import re
from collections.abc import Callable

import litellm
from pydantic import ValidationError

from guarded_agent.guardrails.policy_retrieval import PolicyContext
from guarded_agent.state import PolicyVerdict, ProposedAction

POLICY_CHECK_SYSTEM_PROMPT = """
You are a policy compliance checker for a customer service agent. You will be given
a proposed tool call and the relevant policy clauses. Decide whether the action
should be ALLOWED, DENIED, or requires the user's explicit confirmation first.

Rules:
- ALLOW: the action is clearly permitted by the policy clauses given, and no further
  confirmation is required by those clauses.
- DENY: the action violates the policy clauses given, or is not covered by them.
- NEEDS_CONFIRMATION: the action would be permitted, but the policy requires explicit
  user confirmation before it proceeds, and nothing in the proposed call indicates
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


def _build_user_prompt(action: ProposedAction, context: PolicyContext) -> str:
    return (
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


PolicyCheckFn = Callable[[ProposedAction, PolicyContext], PolicyVerdict]
"""(proposed action, retrieved policy context) -> PolicyVerdict.

Injectable so tests can supply a fixed stub or a fixture-recorded response
instead of a real LLM call, matching graph.py's GenerateFn pattern.
"""


def make_llm_policy_check_fn(model: str, temperature: float = 0.0) -> PolicyCheckFn:
    """Real policy checking via a direct litellm call.

    Deliberately not routed through tau2.utils.llm_utils.generate() like
    graph.py's agent node: this doesn't need tau2's tool-calling message
    format at all (it's evaluating an already-proposed action, not
    proposing one), so keeping it on plain litellm keeps this guardrail
    module independent of tau2, consistent with the rest of guardrails/.
    """

    def _check(action: ProposedAction, context: PolicyContext) -> PolicyVerdict:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": POLICY_CHECK_SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(action, context)},
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
    action: ProposedAction, context: PolicyContext, check_fn: PolicyCheckFn
) -> PolicyVerdict:
    """Thin entry point so callers don't need to know check_fn exists as a
    concept -- mirrors ToolRegistry.dispatch's shape (validated inputs in,
    structured result out)."""
    return check_fn(action, context)
