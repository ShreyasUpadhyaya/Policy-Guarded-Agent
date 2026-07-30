from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from guarded_agent.config import BudgetConfig
from guarded_agent.guardrails.budgets import check_budget
from guarded_agent.state import AgentState, Message, ProposedAction, ToolCall
from guarded_agent.tools.registry import ToolDefinition, ToolRegistry

DEFAULT_BUDGET_LIMITS = BudgetConfig(
    max_steps=1000, max_tool_calls=1000, max_tokens=10_000_000, max_wall_clock_seconds=3600
)
"""Generous fallback so tests/callers that don't care about budgets don't
need to pass one. Real limits come from config.py's BudgetConfig (commit 4),
loaded by whoever actually cares -- currently the tau2 adapter."""


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
    limits: BudgetConfig,
) -> Callable[[AgentState], Literal["executor", "agent", "kill_switch"]]:
    """Wrap `router` with a budget check that runs first.

    Kept separate from `router` itself so `router`'s own tests (which
    exercise the proposed_action routing logic in isolation) don't need a
    BudgetConfig at all.
    """

    def entry_router(state: AgentState) -> Literal["executor", "agent", "kill_switch"]:
        if check_budget(state.budget, limits).breached:
            return "kill_switch"
        return router(state)

    return entry_router


def make_kill_switch_node(limits: BudgetConfig) -> Callable[[AgentState], dict[str, Any]]:
    """Graceful termination on budget breach: escalate and hand off, rather
    than continuing to call the LLM or dispatch tools."""

    def kill_switch(state: AgentState) -> dict[str, Any]:
        verdict = check_budget(state.budget, limits)
        reason = verdict.reason or "budget exceeded"
        updated = state.add_message(
            Message(
                role="assistant",
                content=(
                    "I'm not able to continue with this request right now -- "
                    "I've hit an internal limit and need to hand this off to a human."
                ),
            )
        )
        updated = updated.escalate(reason)
        return {
            "conversation": updated.conversation,
            "escalated": updated.escalated,
            "escalation_reason": updated.escalation_reason,
        }

    return kill_switch


def make_executor_node(registry: ToolRegistry) -> Callable[[AgentState], dict[str, Any]]:
    """Build the executor node, bound to a specific tool registry.

    Dispatches the pending proposed_action through the registry and records
    the result as a tool message. Only reachable when proposed_action is
    pre-seeded (see router) -- when tau2 drives the graph, tool execution
    happens externally in tau2's own orchestrator, not here.
    """

    def executor(state: AgentState) -> dict[str, Any]:
        action = state.proposed_action
        if action is None:
            raise ValueError("executor node reached with no proposed_action set")

        def placeholder_handler(arguments: dict[str, Any]) -> Any:
            return {"note": f"executed {action.tool_name} (placeholder)"}

        result = registry.dispatch(action.tool_name, action.arguments, placeholder_handler)
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
    append a text reply."""

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

        updated = state.add_message(Message(role="assistant", content=decision.content))
        return {"conversation": updated.conversation, "budget": budget}

    return agent


def build_graph(
    generate_fn: GenerateFn,
    registry: ToolRegistry | None = None,
    budget_limits: BudgetConfig = DEFAULT_BUDGET_LIMITS,
) -> CompiledStateGraph[AgentState]:
    registry = registry or ToolRegistry.load()

    graph = StateGraph(AgentState)
    graph.add_node("executor", make_executor_node(registry))  # type: ignore[call-overload]
    graph.add_node("agent", make_agent_node(generate_fn, registry))  # type: ignore[call-overload]
    graph.add_node("kill_switch", make_kill_switch_node(budget_limits))  # type: ignore[call-overload]
    graph.add_conditional_edges(
        START,
        make_entry_router(budget_limits),
        {"executor": "executor", "agent": "agent", "kill_switch": "kill_switch"},
    )
    graph.add_edge("executor", "agent")
    graph.add_edge("agent", END)
    graph.add_edge("kill_switch", END)

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
