from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from guarded_agent.state import AgentState, Message
from guarded_agent.tools.registry import ToolRegistry


def router(state: AgentState) -> Literal["executor", "respond"]:
    """Route to the executor if an action is pending, otherwise respond directly.

    Purely mechanical for now: nothing populates `proposed_action` yet, since
    the LLM-driven decision logic that would set it arrives with the tau2
    adapter (PLAN.md commit 11). This node exists to prove the routing wiring
    itself works.
    """
    return "executor" if state.proposed_action is not None else "respond"


def make_executor_node(registry: ToolRegistry) -> Callable[[AgentState], dict[str, Any]]:
    """Build the executor node, bound to a specific tool registry.

    Dispatches the pending proposed_action through the registry and records
    the result as a tool message. No real domain tool handlers exist yet, so
    the handler here is a placeholder that commit 11's tau2 adapter replaces.
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
        updated = state.add_message(Message(role="tool", content=content))

        return {
            "conversation": updated.conversation,
            "proposed_action": None,
            "budget": state.budget.increment(tool_calls=1),
        }

    return executor


def respond(state: AgentState) -> dict[str, Any]:
    """Placeholder closing turn.

    Replaced by real LLM generation once the tau2 adapter (commit 11) drives
    this graph with a live model.
    """
    updated = state.add_message(
        Message(role="assistant", content="Is there anything else I can help you with?")
    )
    return {"conversation": updated.conversation}


def build_graph(registry: ToolRegistry | None = None) -> CompiledStateGraph[AgentState]:
    registry = registry or ToolRegistry.load()

    graph = StateGraph(AgentState)
    graph.add_node("executor", make_executor_node(registry))  # type: ignore[call-overload]
    graph.add_node("respond", respond)
    graph.add_conditional_edges(START, router, {"executor": "executor", "respond": "respond"})
    graph.add_edge("executor", "respond")
    graph.add_edge("respond", END)

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
