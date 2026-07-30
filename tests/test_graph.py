from __future__ import annotations

import json
from pathlib import Path

import pytest

from guarded_agent.graph import AgentDecision, GenerateFn, build_graph, router, run
from guarded_agent.state import AgentState, Message, ProposedAction, ToolCall
from guarded_agent.tools.registry import ToolDefinition

MOCK_TASKS_PATH = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "tau2-bench"
    / "data"
    / "tau2"
    / "domains"
    / "mock"
    / "tasks.json"
)


def _stub_text(content: str) -> GenerateFn:
    def _generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        return AgentDecision(content=content)

    return _generate


def _stub_tool_call(tool_call: ToolCall) -> GenerateFn:
    def _generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        return AgentDecision(tool_call=tool_call)

    return _generate


def test_router_picks_agent_when_no_proposed_action() -> None:
    state = AgentState()
    assert router(state) == "agent"


def test_router_picks_executor_when_proposed_action_set() -> None:
    state = AgentState(
        proposed_action=ProposedAction(tool_name="get_order_details", arguments={"order_id": "1"})
    )
    assert router(state) == "executor"


def test_agent_only_path_appends_assistant_message() -> None:
    app = build_graph(_stub_text("Is there anything else I can help you with?"))
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert [m.role for m in result.conversation] == ["user", "assistant"]
    assert result.conversation[-1].content == "Is there anything else I can help you with?"
    assert result.budget.tool_calls_used == 0


def test_agent_tool_call_sets_proposed_action_without_running_executor() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    app = build_graph(_stub_tool_call(tool_call))
    state = AgentState(conversation=[Message(role="user", content="Where's my order?")])

    result = run(app, state)

    assert result.proposed_action == ProposedAction(
        id="call_1", tool_name="get_order_details", arguments={"order_id": "1"}
    )
    # the assistant's tool-call proposal is recorded in history (required for
    # correct tool_use/tool_result correlation on the next LLM call), but the
    # executor never ran: no tool *result* message was appended, budget untouched
    assert [m.role for m in result.conversation] == ["user", "assistant"]
    assert result.conversation[-1].tool_calls == [tool_call]
    assert result.budget.tool_calls_used == 0


def test_executor_path_dispatches_tool_and_increments_budget() -> None:
    app = build_graph(_stub_text("Thanks, all set."))
    state = AgentState(
        conversation=[Message(role="user", content="What's the status of my order?")],
        proposed_action=ProposedAction(
            id="call_1", tool_name="get_order_details", arguments={"order_id": "1"}
        ),
    )

    result = run(app, state)

    assert [m.role for m in result.conversation] == ["user", "tool", "assistant"]
    assert result.conversation[1].tool_call_id == "call_1"
    assert result.proposed_action is None
    assert result.budget.tool_calls_used == 1


def test_executor_path_records_unknown_tool_error_without_raising() -> None:
    app = build_graph(_stub_text("Sorry, something went wrong."))
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        proposed_action=ProposedAction(id="call_1", tool_name="delete_everything", arguments={}),
    )

    result = run(app, state)

    tool_message = result.conversation[-2]
    assert tool_message.role == "tool"
    assert "No tool registered" in (tool_message.content or "")


@pytest.mark.skipif(
    not MOCK_TASKS_PATH.exists(),
    reason="vendored tau2-bench data not present -- run `make install` first",
)
def test_mock_domain_smoke_run_completes() -> None:
    """Seeded from a real tau2 mock-domain task, not a synthetic string.

    Uses a stub generate_fn -- no live LLM call in the test suite, per
    CLAUDE.md's fixture-based testing convention. The real integration is
    verified live via `make smoke-guarded` (PLAN.md commit 11), not here.
    """
    tasks = json.loads(MOCK_TASKS_PATH.read_text(encoding="utf-8"))
    ticket = tasks[0]["ticket"]

    app = build_graph(_stub_text("Done -- I've created that task for you."))
    state = AgentState(conversation=[Message(role="user", content=ticket)])

    result = run(app, state)

    assert result.conversation[-1].role == "assistant"
    assert result.escalated is False
