from __future__ import annotations

import json
import re
from pathlib import Path

from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from guarded_agent.graph import AgentDecision, GenerateFn, build_graph
from guarded_agent.graph import run as run_graph
from guarded_agent.state import AgentState, Message, ToolCall
from guarded_agent.telemetry.tracing import configure_tracing, get_tracer
from guarded_agent.tools.registry import ToolDefinition

TRACE_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "traces" / "sample_trace.json"

EMAIL_PATTERN = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
API_KEY_PATTERN = re.compile(r"\b(sk-[A-Za-z0-9-]{10,}|toolu_[A-Za-z0-9]{10,})\b")


def redact(text: str) -> str:
    """Strip email addresses and API-key-shaped strings from trace text.

    Applied to every span before writing the committed sample trace, per
    CLAUDE.md rule 7. This example is entirely scripted (see main() below),
    so there's nothing real to redact -- but the mechanism itself is real
    and would catch either pattern if present.
    """
    text = EMAIL_PATTERN.sub("[REDACTED_EMAIL]", text)
    return API_KEY_PATTERN.sub("[REDACTED_KEY]", text)


def _demo_generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
    """Deterministic stand-in for a real LLM call.

    This script demonstrates tracing infrastructure, not agent behavior --
    a stub keeps it free, fast, and reproducible rather than spending money
    on a live call for what's fundamentally an infra demo (commit 13).
    """
    if conversation and conversation[-1].role == "tool":
        return AgentDecision(
            content=(
                "Your order is currently in transit and should arrive within 2 business days."
            ),
            prompt_tokens=180,
            completion_tokens=24,
        )
    return AgentDecision(
        tool_call=ToolCall(
            id="call_demo_1", name="get_order_details", arguments={"order_id": "A1234"}
        ),
        prompt_tokens=210,
        completion_tokens=18,
    )


def build_demo_generate_fn() -> GenerateFn:
    return _demo_generate


def main() -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter)

    app = build_graph(build_demo_generate_fn())
    state = AgentState(conversation=[Message(role="user", content="Where is my order?")])

    # Two runs, chained: the first lets the agent propose a tool call; the
    # second (fed that same state, proposed_action still set since nothing
    # here clears it the way the adapter does) reaches the executor, then
    # the agent again to respond given the tool result. Together they walk
    # every node this graph has: agent, executor, agent. Wrapped in one
    # parent span so the two graph.invoke() calls -- each of which starts
    # its own trace context otherwise -- show up as a single conversation.
    with get_tracer().start_as_current_span("conversation"):
        state = run_graph(app, state)
        state = run_graph(app, state)

    spans = exporter.get_finished_spans()
    trace_data = [json.loads(redact(span.to_json())) for span in spans]

    TRACE_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    TRACE_OUTPUT_PATH.write_text(json.dumps(trace_data, indent=2))
    print(f"Wrote {len(trace_data)} spans to {TRACE_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
