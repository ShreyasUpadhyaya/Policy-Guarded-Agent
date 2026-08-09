from __future__ import annotations

import re
from collections.abc import Callable

import litellm
from pydantic import BaseModel

from analysis.trace_loader import Trace
from guarded_agent.guardrails.write_gate import is_affirmative
from guarded_agent.state import Message

NATURAL_TERMINATIONS = {"user_stop", "agent_stop"}
BUDGET_TERMINATIONS = {"max_steps", "timeout", "too_many_errors", "context_window_exceeded"}


def check_schema_violation(trace: Trace) -> bool:
    """A tool call was rejected for invalid arguments.

    Detected via the registry's own error message wording
    (tools/registry.py's ToolError: "Arguments for '<tool>' failed schema
    validation."), which is stable and framework-authored -- not phrased by
    an LLM -- so it's a reliable substring match regardless of which agent
    produced the trace.
    """
    return any(
        step.role == "assistant" and step.content and "failed schema validation" in step.content
        for step in trace.steps
    )


def check_budget_breach(trace: Trace) -> bool:
    """The simulation ended because a resource cap (steps, wall-clock,
    error count) was hit, not because the conversation reached a natural
    stop. Uses tau2's own termination_reason as ground truth rather than
    our own escalation text, which wouldn't exist for a baseline/other
    agent's trace.
    """
    return trace.termination_reason in BUDGET_TERMINATIONS


def check_non_termination(trace: Trace) -> bool:
    """The simulation didn't reach a clean user- or agent-initiated stop."""
    return trace.termination_reason not in NATURAL_TERMINATIONS


def check_missing_confirmation(trace: Trace) -> bool:
    """A write-type tool call that actually matched the task's expected
    action happened without an explicit prior user affirmative.

    Reuses guardrails/write_gate.py's own is_affirmative -- the same
    "yes/proceed/confirm" detection our write gate itself uses -- so this
    catches exactly what our own write gate is supposed to prevent, for any
    trace regardless of which agent produced it: a baseline trace with no
    write gate at all should trip this; a fully-guarded trace should not.
    Uses reward_info.action_checks' tool_type (tau2's own ground truth) to
    identify which calls were writes, since the raw transcript carries no
    per-message read/write tag.
    """
    write_tool_names = {
        check.name
        for check in trace.action_checks
        if check.tool_type == "write" and check.action_match
    }
    if not write_tool_names:
        return False

    for index, step in enumerate(trace.steps):
        if step.role != "assistant" or not step.tool_calls:
            continue
        if not any(call.name in write_tool_names for call in step.tool_calls):
            continue

        preceding_user_steps = [s for s in trace.steps[:index] if s.role == "user"]
        if not preceding_user_steps:
            return True
        last_user_step = preceding_user_steps[-1]
        if not is_affirmative(Message(role="user", content=last_user_step.content)):
            return True
    return False


DETERMINISTIC_CHECKS: dict[str, Callable[[Trace], bool]] = {
    "schema_violation": check_schema_violation,
    "budget_breach": check_budget_breach,
    "non_termination": check_non_termination,
    "missing_confirmation": check_missing_confirmation,
}


def run_deterministic_checks(trace: Trace) -> list[str]:
    """Every deterministic label that fires for this trace -- multi-label,
    not mutually exclusive (a trace can be both non_termination and
    budget_breach at once)."""
    return [name for name, check in DETERMINISTIC_CHECKS.items() if check(trace)]


# --- LLM judge for the residual (PLAN.md commit 27/28) ----------------------

JUDGE_SYSTEM_PROMPT = """
You are labeling why a customer-service agent's conversation failed its task, using
the MAST failure taxonomy (Cemri et al., "Why Do Multi-Agent LLM Systems Fail?").
None of the deterministic checks (schema violation, budget breach, non-termination,
missing confirmation) fired for this trace, so the cause is something subtler in the
conversation itself.

Given the transcript, pick exactly one of these 14 modes that best explains the
failure:

FM-1.1 Disobey task specification, FM-1.2 Disobey role specification,
FM-1.3 Step repetition, FM-1.4 Loss of conversation history,
FM-1.5 Unaware of termination conditions, FM-2.1 Conversation reset,
FM-2.2 Fail to ask for clarification, FM-2.3 Task derailment,
FM-2.4 Information withholding, FM-2.5 Ignored other agent's input,
FM-2.6 Reasoning-action mismatch, FM-3.1 Premature termination,
FM-3.2 No or incomplete verification, FM-3.3 Incorrect verification.

Respond with ONLY a JSON object, no other text, no markdown code fences, with
exactly these keys: "label" (one of the FM-x.x codes above), "reason" (one sentence).
""".strip()

_JSON_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


class JudgeVerdict(BaseModel):
    label: str
    reason: str


def _extract_json_object(text: str) -> str:
    """Same defensive extraction as guardrails/policy_checker.py and
    guardrails/critic.py -- kept as a separate small copy for the same
    reason: each module stays independently readable."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
    match = _JSON_OBJECT_PATTERN.search(stripped)
    return match.group(0) if match else stripped


def _format_transcript(trace: Trace) -> str:
    lines = []
    for step in trace.steps:
        if step.role == "tool":
            lines.append(f"tool result: {step.content}")
        elif step.tool_calls:
            calls = ", ".join(f"{c.name}({c.arguments})" for c in step.tool_calls)
            lines.append(f"{step.role} (tool call): {calls}")
        elif step.content:
            lines.append(f"{step.role}: {step.content}")
    return "\n".join(lines)


JudgeFn = Callable[[Trace], JudgeVerdict]
"""trace -> JudgeVerdict. Injectable so tests can supply a fixed stub or a
fixture-recorded response instead of a real LLM call, matching graph.py's
GenerateFn and the guardrails' *CheckFn pattern.
"""


def make_llm_judge_fn(model: str, temperature: float = 0.0) -> JudgeFn:
    """Real judging via a direct litellm call, independent of tau2 for the
    same reason guardrails/policy_checker.py and guardrails/critic.py are:
    this evaluates an already-finished transcript, not tau2's live
    tool-calling format.

    A parse failure returns label="unknown" rather than guessing at one of
    the 14 real modes -- an unlabeled trace is honest; a wrong label
    silently corrupts commit 28's hand-label agreement measurement.
    """

    def _judge(trace: Trace) -> JudgeVerdict:
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": _format_transcript(trace)},
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        if content is None:
            return JudgeVerdict(label="unknown", reason="judge LLM call returned no content")

        json_text = _extract_json_object(content)
        try:
            return JudgeVerdict.model_validate_json(json_text)
        except Exception as exc:  # boundary around an LLM's free-text response
            return JudgeVerdict(label="unknown", reason=f"judge response did not parse: {exc}")

    return _judge


class TraceLabel(BaseModel):
    """The full label for one failed trace: deterministic hits (possibly
    several, possibly none) plus an optional judge verdict."""

    trace_id: str
    task_id: str
    deterministic_labels: list[str]
    judge_label: str | None = None
    judge_reason: str | None = None


def label_trace(trace: Trace, judge_fn: JudgeFn | None = None) -> TraceLabel:
    """Label one failed trace: deterministic checks run first and are
    authoritative when any fire. The LLM judge (if provided) only runs on
    the residual -- a trace where nothing deterministic explains the
    failure -- per PLAN.md commit 27: "deterministic checks first... LLM
    judge only for the residual."
    """
    deterministic = run_deterministic_checks(trace)
    judge_verdict = judge_fn(trace) if not deterministic and judge_fn is not None else None
    return TraceLabel(
        trace_id=trace.id,
        task_id=trace.task_id,
        deterministic_labels=deterministic,
        judge_label=judge_verdict.label if judge_verdict else None,
        judge_reason=judge_verdict.reason if judge_verdict else None,
    )


def label_failed_traces(traces: list[Trace], judge_fn: JudgeFn | None = None) -> list[TraceLabel]:
    """Label every failed trace (reward < 1.0) in a list -- successful
    traces have nothing to diagnose."""
    return [label_trace(trace, judge_fn) for trace in traces if trace.reward < 1.0]
