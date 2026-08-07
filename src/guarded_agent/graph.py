from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Hashable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from guarded_agent.config import BudgetConfig
from guarded_agent.guardrails.budgets import check_budget
from guarded_agent.guardrails.critic import CriticCheckFn, check_response
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


def make_after_agent_router(
    has_critic: bool,
) -> Callable[[AgentState], Literal["policy_gate", "critic", "end"]]:
    """A proposed tool call must clear the policy gate before it can be sent
    out. A plain text reply either goes to the critic (PLAN.md commit 20,
    if one is configured) or straight out -- except a reply the agent node
    already escalated on (an empty decision, see make_agent_node) skips the
    critic entirely, since there is no drafted response left to review, only
    the hand-off message the agent node already appended.
    """

    def after_agent_router(state: AgentState) -> Literal["policy_gate", "critic", "end"]:
        if state.proposed_action is not None:
            return "policy_gate"
        if state.escalated or not has_critic:
            return "end"
        return "critic"

    return after_agent_router


def after_policy_gate_router(state: AgentState) -> Literal["write_gate", "end"]:
    """The policy gate clears proposed_action itself on denial/invalid
    schema -- if it's still set, the action was allowed through to the
    write gate; if not, the turn already has its answer."""
    return "write_gate" if state.proposed_action is not None else "end"


def after_critic_router(state: AgentState) -> Literal["agent_revise", "end"]:
    """A REVISE verdict leaves critic_feedback set (and the rejected draft
    already popped off conversation, see make_critic_node) -- route to the
    one bounded revision. An APPROVE verdict is a no-op: the draft already
    sits in conversation, ready to send."""
    return "agent_revise" if state.critic_feedback is not None else "end"


def after_agent_revise_router(state: AgentState) -> Literal["policy_gate", "critic_final", "end"]:
    """A revision that proposes a tool call instead of text is a legitimate
    outcome, not a failure -- the critic rejecting an ungrounded claim is
    exactly what should push the model toward looking the answer up instead
    of asserting it, and that call still has to clear policy_gate/write_gate
    like any other proposed action. agent_revise escalates directly (ending
    the turn there) only if the revision produced neither text nor a tool
    call -- nothing usable to route anywhere. Otherwise the revised text
    draft gets exactly one more critic pass."""
    if state.proposed_action is not None:
        return "policy_gate"
    return "end" if state.escalated else "critic_final"


def after_critic_final_router(state: AgentState) -> Literal["escalation", "end"]:
    """The bounded end of the critic loop (PLAN.md commit 20: "at most one
    revision loop, then it must proceed or escalate"). A second REVISE
    verdict here never leads back to agent_revise -- only to escalation --
    which is what makes the loop structurally bounded rather than merely
    counted."""
    return "escalation" if state.critic_feedback is not None else "end"


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
    escalation was needed, not *why*) to report an accurate reason -- now
    including a second critic rejection (PLAN.md commit 20), the one
    trigger this node can't detect from state.budget/conversation alone.
    If the domain registers a transfer_to_human_agents tool, proposes that
    call so tau2 records a real transfer instead of just a text message
    saying so; the proposal bypasses policy_gate/write_gate deliberately --
    routing an escalation caused by policy deadlock back through the policy
    gate risks another denial, defeating the point of escalating.
    """

    def escalation(state: AgentState) -> dict[str, Any]:
        budget_verdict = check_budget(state.budget, budget_limits)
        failure_verdict = check_repeated_tool_failure(
            state.conversation, max_consecutive_tool_failures
        )
        deadlock_verdict = check_policy_deadlock(
            state.consecutive_policy_denials, max_consecutive_policy_denials
        )
        critic_reason = (
            f"critic rejected the revised response: {state.critic_feedback}"
            if state.critic_feedback
            else None
        )
        reason = (
            budget_verdict.reason
            or failure_verdict.reason
            or deadlock_verdict.reason
            or critic_reason
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
                "critic_feedback": None,
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
            "critic_feedback": None,
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


def _reject_proposal(state: AgentState, explanation: str) -> AgentState:
    """Replace a just-proposed, never-sent, never-executed tool-call message
    with a plain-text explanation.

    The proposal must be removed, not just left alongside the explanation --
    verified live during commit 21's v2 smoke run: leaving it produced a
    tool_use block with no following tool_result (tau2 never dispatches a
    call the gate rejected), and Anthropic rejects the *next* LLM call
    outright for it ("tool_use ids were found without tool_result blocks").
    Every node that can reject an already-proposed action -- policy_gate's
    schema/DENY branches, write_gate's not-yet-confirmed branch -- must call
    this instead of state.add_message so the rejected proposal never lingers
    in history as if it had gone out for real.
    """
    popped = state.model_copy(update={"conversation": state.conversation[:-1]})
    return popped.add_message(Message(role="assistant", content=explanation))


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
            updated = _reject_proposal(state, message)
            return {"conversation": updated.conversation, "proposed_action": None}

        query = _build_retrieval_query(action)
        context = get_policy_context(retriever, query, full_policy_text, top_k, min_confidence)
        verdict = check_policy(action, context, policy_check_fn)

        if verdict.verdict == "DENY":
            updated = _reject_proposal(state, f"I can't do that: {verdict.reason}")
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

        assert gate_verdict.prompt_message is not None  # always set when confirmed=False
        updated = _reject_proposal(state, gate_verdict.prompt_message)
        return {
            "conversation": updated.conversation,
            "proposed_action": None,
            "pending_confirmation": action,
        }

    return write_gate


def make_critic_node(
    critic_check_fn: CriticCheckFn,
    retriever: PolicyRetriever,
    full_policy_text: str,
    top_k: int,
    min_confidence: float,
) -> Callable[[AgentState], dict[str, Any]]:
    """Review a drafted text response for unsupported claims and policy
    drift (PLAN.md commit 20) before it goes out. Only reachable for text
    replies -- tool-call proposals are already gated by policy_gate/write_gate
    and never pass through here (see make_after_agent_router).

    Used for both the first and second (final) critic pass -- the same
    node function is added to the graph under two different node names,
    each wired to a different router, so the bounded-retry guarantee comes
    from the graph's topology (see build_graph) rather than a counter this
    node would have to remember to check.

    An APPROVE verdict is a no-op: the draft already sits in conversation,
    ready to send. A REVISE verdict pops the rejected draft back off
    conversation -- the user never saw it -- and records the critic's
    reason in critic_feedback for the next node (agent_revise, or
    escalation on the second pass) to consume.
    """

    def critic(state: AgentState) -> dict[str, Any]:
        draft = state.conversation[-1]
        if draft.role != "assistant" or not draft.content:
            raise ValueError("critic node reached without a drafted text response")

        context = get_policy_context(
            retriever, draft.content, full_policy_text, top_k, min_confidence
        )
        verdict = check_response(state.conversation[:-1], draft.content, context, critic_check_fn)
        if verdict.approved:
            return {}

        return {"conversation": state.conversation[:-1], "critic_feedback": verdict.reason}

    return critic


def make_agent_revise_node(
    generate_fn: GenerateFn, registry: ToolRegistry
) -> Callable[[AgentState], dict[str, Any]]:
    """The critic's one bounded revision (PLAN.md commit 20). Regenerates a
    response with the critic's feedback appended as a one-off note -- never
    persisted into conversation itself, only used to steer this single
    regeneration call -- then always clears critic_feedback, so nothing
    about this pass can linger into a state a future turn might see.

    A revision that proposes a tool call is a legitimate outcome, not a
    failure: the critic rejecting an ungrounded claim is exactly what
    should push the model toward looking the answer up instead of
    asserting it (verified live -- an earlier version of this node treated
    any tool call here as unusable and escalated immediately, which turned
    out to tank a live smoke run's Pass^1 to 0.000 by escalating on
    completely ordinary "let me check that" recoveries). That call still
    goes through policy_gate/write_gate like any other proposed action (see
    after_agent_revise_router). Only a decision with neither text nor a
    tool call -- nothing usable at all -- escalates immediately, the same
    fail-safe-forcing rule make_agent_node applies to an empty first-pass
    decision.
    """

    def agent_revise(state: AgentState) -> dict[str, Any]:
        revision_note = (
            "[Internal note: your previous draft was rejected by review -- "
            f"{state.critic_feedback}. Provide a corrected response addressing this.]"
        )
        revision_conversation = [*state.conversation, Message(role="user", content=revision_note)]
        decision = generate_fn(revision_conversation, list(registry.tools.values()))
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
                "critic_feedback": None,
            }

        if not decision.content:
            escalated = state.escalate("revised response was not usable after one critic revision")
            escalated = escalated.add_message(
                Message(
                    role="assistant",
                    content=(
                        "I'm not able to give you a reliable answer right now -- "
                        "handing this off to a human."
                    ),
                )
            )
            return {
                "conversation": escalated.conversation,
                "escalated": escalated.escalated,
                "escalation_reason": escalated.escalation_reason,
                "budget": budget,
                "critic_feedback": None,
            }

        updated = state.add_message(Message(role="assistant", content=decision.content))
        return {"conversation": updated.conversation, "budget": budget, "critic_feedback": None}

    return agent_revise


def build_graph(
    generate_fn: GenerateFn,
    registry: ToolRegistry | None = None,
    budget_limits: BudgetConfig = DEFAULT_BUDGET_LIMITS,
    policy_check_fn: PolicyCheckFn | None = None,
    retriever: PolicyRetriever | None = None,
    full_policy_text: str = "",
    retrieval_top_k: int = 3,
    retrieval_min_confidence: float = 0.3,
    critic_check_fn: CriticCheckFn | None = None,
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

        after_agent_mapping: dict[Hashable, str] = {"policy_gate": "policy_gate", "end": END}
        if critic_check_fn is not None:
            after_agent_mapping["critic"] = "critic"
        graph.add_conditional_edges(
            "agent", make_after_agent_router(critic_check_fn is not None), after_agent_mapping
        )
        graph.add_conditional_edges(
            "policy_gate", after_policy_gate_router, {"write_gate": "write_gate", "end": END}
        )
        graph.add_edge("write_gate", END)

        if critic_check_fn is not None:
            critic_node_fn = make_critic_node(
                critic_check_fn,
                retriever,
                full_policy_text,
                retrieval_top_k,
                retrieval_min_confidence,
            )
            graph.add_node("critic", traced_node("critic", critic_node_fn))  # type: ignore[call-overload]
            graph.add_conditional_edges(
                "critic", after_critic_router, {"agent_revise": "agent_revise", "end": END}
            )

            graph.add_node(
                "agent_revise",
                traced_node("agent_revise", make_agent_revise_node(generate_fn, registry)),
            )  # type: ignore[call-overload]
            graph.add_conditional_edges(
                "agent_revise",
                after_agent_revise_router,
                {"policy_gate": "policy_gate", "critic_final": "critic_final", "end": END},
            )

            graph.add_node("critic_final", traced_node("critic_final", critic_node_fn))  # type: ignore[call-overload]
            graph.add_conditional_edges(
                "critic_final",
                after_critic_final_router,
                {"escalation": "escalation", "end": END},
            )
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
