from __future__ import annotations

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from guarded_agent.graph import AgentDecision, build_graph, run
from guarded_agent.state import AgentState, Message, ProposedAction, ToolCall
from guarded_agent.telemetry.tracing import configure_tracing


def _stub_tool_call(conversation: list[Message], tools: object) -> AgentDecision:
    return AgentDecision(
        tool_call=ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    )


def test_executor_and_agent_nodes_emit_spans_with_expected_attributes() -> None:
    # configure_tracing() is idempotent (only its first call across the whole
    # process takes effect) -- this is currently the only place that calls
    # it, so it's safe as-is. If a later commit adds another caller, this
    # test's exporter would silently stop receiving spans.
    exporter = InMemorySpanExporter()
    configure_tracing(exporter)

    app = build_graph(_stub_tool_call)
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        proposed_action=ProposedAction(
            id="call_1", tool_name="get_order_details", arguments={"order_id": "1"}
        ),
    )
    run(app, state)

    spans = {span.name: span for span in exporter.get_finished_spans()}

    assert "node.executor" in spans
    assert "node.agent" in spans
    assert "tool.get_order_details" in spans

    tool_span = spans["tool.get_order_details"]
    assert tool_span.attributes is not None
    assert tool_span.attributes["guarded_agent.tool.name"] == "get_order_details"
    assert tool_span.attributes["guarded_agent.tool.ok"] is True

    executor_span = spans["node.executor"]
    assert executor_span.attributes is not None
    assert executor_span.attributes["guarded_agent.node"] == "executor"
