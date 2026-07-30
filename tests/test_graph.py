from __future__ import annotations

import json
from pathlib import Path

import pytest

from guarded_agent.graph import build_graph, router, run
from guarded_agent.state import AgentState, Message, ProposedAction

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


def test_router_picks_respond_when_no_proposed_action() -> None:
    state = AgentState()
    assert router(state) == "respond"


def test_router_picks_executor_when_proposed_action_set() -> None:
    state = AgentState(
        proposed_action=ProposedAction(tool_name="get_order_details", arguments={"order_id": "1"})
    )
    assert router(state) == "executor"


def test_respond_only_path_appends_assistant_message() -> None:
    app = build_graph()
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert [m.role for m in result.conversation] == ["user", "assistant"]
    assert result.budget.tool_calls_used == 0


def test_executor_path_dispatches_tool_and_increments_budget() -> None:
    app = build_graph()
    state = AgentState(
        conversation=[Message(role="user", content="What's the status of my order?")],
        proposed_action=ProposedAction(tool_name="get_order_details", arguments={"order_id": "1"}),
    )

    result = run(app, state)

    assert [m.role for m in result.conversation] == ["user", "tool", "assistant"]
    assert result.proposed_action is None
    assert result.budget.tool_calls_used == 1


def test_executor_path_records_unknown_tool_error_without_raising() -> None:
    app = build_graph()
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        proposed_action=ProposedAction(tool_name="delete_everything", arguments={}),
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
    """Seeded from a real tau2 mock-domain task, not a synthetic string."""
    tasks = json.loads(MOCK_TASKS_PATH.read_text(encoding="utf-8"))
    ticket = tasks[0]["ticket"]

    app = build_graph()
    state = AgentState(conversation=[Message(role="user", content=ticket)])

    result = run(app, state)

    assert result.conversation[-1].role == "assistant"
    assert result.escalated is False
