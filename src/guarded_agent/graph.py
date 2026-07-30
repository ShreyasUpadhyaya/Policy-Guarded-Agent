from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel

from guarded_agent.state import AgentState, Message, ProposedAction, ToolCall
from guarded_agent.tools.registry import ToolDefinition, ToolRegistry


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
        budget = state.budget.increment(tokens=decision.prompt_tokens + decision.completion_tokens)

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
) -> CompiledStateGraph[AgentState]:
    registry = registry or ToolRegistry.load()

    graph = StateGraph(AgentState)
    graph.add_node("executor", make_executor_node(registry))  # type: ignore[call-overload]
    graph.add_node("agent", make_agent_node(generate_fn, registry))  # type: ignore[call-overload]
    graph.add_conditional_edges(START, router, {"executor": "executor", "agent": "agent"})
    graph.add_edge("executor", "agent")
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
