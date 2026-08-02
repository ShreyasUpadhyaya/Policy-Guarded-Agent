from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from guarded_agent.config import BudgetConfig
from guarded_agent.guardrails.budgets import check_budget
from guarded_agent.guardrails.escalation import check_policy_deadlock, check_repeated_tool_failure
from guarded_agent.guardrails.policy_checker import PolicyCheckFn, check_policy
from guarded_agent.guardrails.policy_retrieval import PolicyRetriever, get_policy_context
from guarded_agent.guardrails.write_gate import DispatchCache, evaluate_write_gate
from guarded_agent.state import AgentState, Message, ProposedAction, ToolCall
from guarded_agent.telemetry.tracing import traced_node, traced_tool_call
from guarded_agent.tools.registry import ToolDefinition, ToolRegistry

DEFAULT_BUDGET_LIMITS = BudgetConfig(
    max_steps=1000, max_tool_calls=1000, max_tokens=10_000_000, max_wall_clock_seconds=3600
)
"""Generous fallback so tests/callers that don't care about budgets don't
need to pass one. Real limits come from config.py's BudgetConfig (commit 4),
loaded by whoever actually cares -- currently the tau2 adapter."""

DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES = 1000
DEFAULT_MAX_CONSECUTIVE_POLICY_DENIALS = 1000
"""Same rationale as DEFAULT_BUDGET_LIMITS: generous fallbacks so callers
that don't care about escalation thresholds don't need to pass them."""

TRANSFER_TOOL_NAME = "transfer_to_human_agents"
"""tau2's own convention for a human handoff tool, named explicitly in
several domains' policies (including retail's). When the registry has it,
escalation proposes a real tool call instead of just saying so in text."""


class AgentDecision(BaseModel):
    """What the agent node's LLM call decided to do this turn."""

    content: str | None = None
    tool_call: ToolCall | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0


GenerateFn = Callable[[list[Message], list[ToolDefinition]], AgentDecision]
"""(conversation, available tools) -> AgentDecision.

Injectable so tests can supply a fixed stub instead of a real LLM call,
per CLAUDE.md: "LLM-dependent behaviour gets fixture-based tests... no
live API calls in the test suite." The real implementation, wired in by
the tau2 adapter (commit 11), reuses tau2.utils.llm_utils.generate rather
than reimplementing tool-call formatting from scratch.
"""


def router(state: AgentState) -> Literal["executor", "agent"]:
    """Route to the executor if an action is already pending, otherwise let
    the agent decide. Nothing pre-seeds `proposed_action` in the tau2-driven
    path, so that path always reaches `agent`; a pre-seeded proposed_action
    (as in some tests, or a hypothetical non-benchmarked deployment) reaches
    `executor` directly.
    """
    return "executor" if state.proposed_action is not None else "agent"


def make_entry_router(
    budget_limits: BudgetConfig,
    max_consecutive_tool_failures: int = DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES,
    max_consecutive_policy_denials: int = DEFAULT_MAX_CONSECUTIVE_POLICY_DENIALS,
) -> Callable[[AgentState], Literal["executor", "agent", "escalation"]]:
    """Wrap `router` with the three escalation triggers (PLAN.md commit 18),
    checked before anything else runs this turn.

    Kept separate from `router` itself so `router`'s own tests (which
    exercise the proposed_action routing logic in isolation) don't need any
    of these configs at all.
    """

    def entry_router(state: AgentState) -> Literal["executor", "agent", "escalation"]:
        if check_budget(state.budget, budget_limits).breached:
            return "escalation"
        failure_check = check_repeated_tool_failure(
            state.conversation, max_consecutive_tool_failures
        )
        if failure_check.should_escalate:
            return "escalation"
        if check_policy_deadlock(
            state.consecutive_policy_denials, max_consecutive_policy_denials
        ).should_escalate:
            return "escalation"
        return router(state)

    return entry_router


def after_agent_router(state: AgentState) -> Literal["policy_gate", "end"]:
    """A proposed tool call must clear the policy gate before it can be sent
    out; a plain text reply needs no policy check."""
    return "policy_gate" if state.proposed_action is not None else "end"


def after_policy_gate_router(state: AgentState) -> Literal["write_gate", "end"]:
    """The policy gate clears proposed_action itself on denial/invalid
    schema -- if it's still set, the action was allowed through to the
    write gate; if not, the turn already has its answer."""
    return "write_gate" if state.proposed_action is not None else "end"


def make_escalation_node(
    registry: ToolRegistry,
    budget_limits: BudgetConfig,
    max_consecutive_tool_failures: int,
    max_consecutive_policy_denials: int,
) -> Callable[[AgentState], dict[str, Any]]:
    """Graceful termination on any of three triggers (budget breach, repeated
    tool failure, policy deadlock -- PLAN.md commit 18): escalate rather
    than continuing to call the LLM or dispatch tools.

    Recomputes which trigger fired (make_entry_router only decided *that*
    escalation was needed, not *why*) to report an accurate reason. If the
    domain registers a transfer_to_human_agents tool, proposes that call so
    tau2 records a real transfer instead of just a text message saying so;
    the proposal bypasses policy_gate/write_gate deliberately -- routing an
    escalation caused by policy deadlock back through the policy gate risks
    another denial, defeating the point of escalating.
    """

    def escalation(state: AgentState) -> dict[str, Any]:
        budget_verdict = check_budget(state.budget, budget_limits)
        failure_verdict = check_repeated_tool_failure(
            state.conversation, max_consecutive_tool_failures
        )
        deadlock_verdict = check_policy_deadlock(
            state.consecutive_policy_denials, max_consecutive_policy_denials
        )
        reason = (
            budget_verdict.reason
            or failure_verdict.reason
            or deadlock_verdict.reason
            or "escalated"
        )
        updated = state.escalate(reason)

        transfer_tool = registry.get(TRANSFER_TOOL_NAME)
        if transfer_tool is not None:
            call = ToolCall(
                id=f"escalation-{uuid.uuid4()}",
                name=TRANSFER_TOOL_NAME,
                arguments={"summary": reason},
            )
            updated = updated.add_message(Message(role="assistant", tool_calls=[call]))
            return {
                "conversation": updated.conversation,
                "escalated": updated.escalated,
                "escalation_reason": updated.escalation_reason,
                "proposed_action": ProposedAction(
                    id=call.id, tool_name=call.name, arguments=call.arguments
                ),
            }

        updated = updated.add_message(
            Message(
                role="assistant",
                content=(
                    "I'm not able to continue with this request right now -- "
                    "I've hit an internal limit and need to hand this off to a human."
                ),
            )
        )
        return {
            "conversation": updated.conversation,
            "escalated": updated.escalated,
            "escalation_reason": updated.escalation_reason,
        }

    return escalation


def make_executor_node(
    registry: ToolRegistry,
    session_id: str,
    dispatch_cache: DispatchCache | None = None,
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the executor node, bound to a specific tool registry.

    Dispatches the pending proposed_action through the registry and records
    the result as a tool message. Only reachable when proposed_action is
    pre-seeded (see router) -- when tau2 drives the graph, tool execution
    happens externally in tau2's own orchestrator, not here.

    Idempotent per (session_id, turn, action_hash): a retried dispatch for
    the same key returns the original result instead of calling the handler
    again (PLAN.md commit 17). `turn` is state.budget.steps_used at dispatch
    time, which is identical across two independent runs started from the
    same state -- exactly what "retried" means here.
    """
    cache = dispatch_cache if dispatch_cache is not None else DispatchCache()

    def executor(state: AgentState) -> dict[str, Any]:
        action = state.proposed_action
        if action is None:
            raise ValueError("executor node reached with no proposed_action set")

        def placeholder_handler(arguments: dict[str, Any]) -> Any:
            return {"note": f"executed {action.tool_name} (placeholder)"}

        def do_dispatch() -> Any:
            with traced_tool_call(action.tool_name, action.arguments) as span:
                dispatch_result = registry.dispatch(
                    action.tool_name, action.arguments, placeholder_handler
                )
                span.set_attribute("guarded_agent.tool.ok", dispatch_result.ok)
                return dispatch_result

        result, _was_cached = cache.get_or_dispatch(
            session_id, state.budget.steps_used, action, do_dispatch
        )
        content = (
            str(result.result)
            if result.ok
            else (result.error.message if result.error else "unknown tool error")
        )
        updated = state.add_message(Message(role="tool", content=content, tool_call_id=action.id))

        return {
            "conversation": updated.conversation,
            "proposed_action": None,
            "budget": state.budget.increment(tool_calls=1),
        }

    return executor


def make_agent_node(
    generate_fn: GenerateFn, registry: ToolRegistry
) -> Callable[[AgentState], dict[str, Any]]:
    """Build the agent node: call the LLM, then either propose an action or
    append a text reply.

    A decision with neither content nor a tool call (the model produced
    nothing usable) fails to escalation rather than sending an empty
    message -- tau2 itself rejects empty assistant messages, and silently
    forwarding one is exactly the kind of degraded input CLAUDE.md says
    must never resolve to normal, silent behaviour.
    """

    def agent(state: AgentState) -> dict[str, Any]:
        decision = generate_fn(state.conversation, list(registry.tools.values()))
        budget = state.budget.increment(
            steps=1, tokens=decision.prompt_tokens + decision.completion_tokens
        )

        if decision.tool_call is not None:
            updated = state.add_message(Message(role="assistant", tool_calls=[decision.tool_call]))
            return {
                "conversation": updated.conversation,
                "proposed_action": ProposedAction(
                    id=decision.tool_call.id,
                    tool_name=decision.tool_call.name,
                    arguments=decision.tool_call.arguments,
                ),
                "budget": budget,
            }

        if not decision.content:
            escalated = state.escalate("agent produced neither a reply nor a tool call")
            escalated = escalated.add_message(
                Message(
                    role="assistant",
                    content="I wasn't able to determine how to respond -- handing off to a human.",
                )
            )
            return {
                "conversation": escalated.conversation,
                "escalated": escalated.escalated,
                "escalation_reason": escalated.escalation_reason,
                "budget": budget,
            }

        updated = state.add_message(Message(role="assistant", content=decision.content))
        return {"conversation": updated.conversation, "budget": budget}

    return agent


def _build_retrieval_query(action: ProposedAction) -> str:
    return f"{action.tool_name} {json.dumps(action.arguments, sort_keys=True)}"


def make_policy_gate_node(
    registry: ToolRegistry,
    retriever: PolicyRetriever,
    policy_check_fn: PolicyCheckFn,
    full_policy_text: str,
    top_k: int,
    min_confidence: float,
) -> Callable[[AgentState], dict[str, Any]]:
    """Validate a proposed action's arguments against its schema (commit 9's
    registry -- previously only exercised in the pre-seeded/standalone path,
    never in the live tau2-driven flow, since tau2 executes tools itself and
    the graph used to end the turn right after `agent`), then check it
    against retrieved policy (commit 15/16). DENY or an invalid schema
    clears proposed_action and answers with an explanation; ALLOW and
    NEEDS_CONFIRMATION pass through to the write gate.
    """

    def policy_gate(state: AgentState) -> dict[str, Any]:
        action = state.proposed_action
        if action is None:
            raise ValueError("policy_gate node reached with no proposed_action set")

        validation = registry.dispatch(action.tool_name, action.arguments, lambda _args: None)
        if not validation.ok:
            message = validation.error.message if validation.error else "invalid tool call"
            updated = state.add_message(Message(role="assistant", content=message))
            return {"conversation": updated.conversation, "proposed_action": None}

        query = _build_retrieval_query(action)
        context = get_policy_context(retriever, query, full_policy_text, top_k, min_confidence)
        verdict = check_policy(action, context, policy_check_fn)

        if verdict.verdict == "DENY":
            updated = state.add_message(
                Message(role="assistant", content=f"I can't do that: {verdict.reason}")
            )
            return {
                "conversation": updated.conversation,
                "proposed_action": None,
                "policy_verdict": verdict,
                "consecutive_policy_denials": state.consecutive_policy_denials + 1,
            }

        return {"policy_verdict": verdict, "consecutive_policy_denials": 0}

    return policy_gate


def make_write_gate_node(registry: ToolRegistry) -> Callable[[AgentState], dict[str, Any]]:
    """Gate mutating actions behind explicit user confirmation.

    Runs after the policy gate, so `state.policy_verdict` is already set.
    A NEEDS_CONFIRMATION verdict gates regardless of the tool's own
    mutating flag; otherwise gating follows the tool's real classification
    (adapters/tau2_agent.py extracts this from tau2's own tool metadata).
    """

    def write_gate(state: AgentState) -> dict[str, Any]:
        action = state.proposed_action
        verdict = state.policy_verdict
        if action is None or verdict is None:
            raise ValueError("write_gate node reached with no proposed_action/policy_verdict set")

        definition = registry.get(action.tool_name)
        needs_gate = verdict.verdict == "NEEDS_CONFIRMATION" or (
            definition is not None and definition.mutating
        )
        if not needs_gate:
            return {}

        last_user_message = next(
            (m for m in reversed(state.conversation) if m.role == "user"), None
        )
        gate_verdict = evaluate_write_gate(
            action, state.pending_confirmation, last_user_message, verdict.clause_id, verdict.reason
        )
        if gate_verdict.confirmed:
            return {"pending_confirmation": None}

        updated = state.add_message(Message(role="assistant", content=gate_verdict.prompt_message))
        return {
            "conversation": updated.conversation,
            "proposed_action": None,
            "pending_confirmation": action,
        }

    return write_gate


def build_graph(
    generate_fn: GenerateFn,
    registry: ToolRegistry | None = None,
    budget_limits: BudgetConfig = DEFAULT_BUDGET_LIMITS,
    policy_check_fn: PolicyCheckFn | None = None,
    retriever: PolicyRetriever | None = None,
    full_policy_text: str = "",
    retrieval_top_k: int = 3,
    retrieval_min_confidence: float = 0.3,
    session_id: str | None = None,
    dispatch_cache: DispatchCache | None = None,
    max_consecutive_tool_failures: int = DEFAULT_MAX_CONSECUTIVE_TOOL_FAILURES,
    max_consecutive_policy_denials: int = DEFAULT_MAX_CONSECUTIVE_POLICY_DENIALS,
) -> CompiledStateGraph[AgentState]:
    registry = registry or ToolRegistry.load()
    session_id = session_id or str(uuid.uuid4())

    graph = StateGraph(AgentState)
    graph.add_node(
        "executor",
        traced_node("executor", make_executor_node(registry, session_id, dispatch_cache)),
    )  # type: ignore[call-overload]
    graph.add_node("agent", traced_node("agent", make_agent_node(generate_fn, registry)))  # type: ignore[call-overload]
    escalation_node = make_escalation_node(
        registry, budget_limits, max_consecutive_tool_failures, max_consecutive_policy_denials
    )
    graph.add_node("escalation", traced_node("escalation", escalation_node))  # type: ignore[call-overload]
    entry_router = make_entry_router(
        budget_limits, max_consecutive_tool_failures, max_consecutive_policy_denials
    )
    graph.add_conditional_edges(
        START,
        entry_router,
        {"executor": "executor", "agent": "agent", "escalation": "escalation"},
    )
    graph.add_edge("executor", "agent")
    graph.add_edge("escalation", END)

    if policy_check_fn is not None and retriever is not None:
        graph.add_node(
            "policy_gate",
            traced_node(
                "policy_gate",
                make_policy_gate_node(
                    registry,
                    retriever,
                    policy_check_fn,
                    full_policy_text,
                    retrieval_top_k,
                    retrieval_min_confidence,
                ),
            ),
        )  # type: ignore[call-overload]
        graph.add_node("write_gate", traced_node("write_gate", make_write_gate_node(registry)))  # type: ignore[call-overload]
        graph.add_conditional_edges(
            "agent", after_agent_router, {"policy_gate": "policy_gate", "end": END}
        )
        graph.add_conditional_edges(
            "policy_gate", after_policy_gate_router, {"write_gate": "write_gate", "end": END}
        )
        graph.add_edge("write_gate", END)
    else:
        # No policy checker wired in (e.g. tests exercising router/executor/agent
        # in isolation, as commit 10 already established) -- a proposed action
        # goes straight out, matching the pre-commit-17 graph shape exactly.
        graph.add_edge("agent", END)

    return graph.compile()


def run(app: CompiledStateGraph[AgentState], state: AgentState) -> AgentState:
    """Invoke the graph once and return a fully-populated AgentState.

    `app.invoke()` returns a dict containing only the schema fields a node
    actually touched during the run -- Optional-typed fields (plan,
    proposed_action, policy_verdict, escalation_reason) silently disappear
    from it if untouched, rather than coming back as None. This normalizes
    that back into a complete, predictable AgentState instance.
    """
    return AgentState.model_validate(app.invoke(state))
