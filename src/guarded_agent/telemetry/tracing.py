from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, SimpleSpanProcessor, SpanExporter
from opentelemetry.trace import Span, Tracer

from guarded_agent.state import AgentState

TRACER_NAME = "guarded_agent"

_configured = False


def configure_tracing(exporter: SpanExporter | None = None) -> None:
    """Set up the global TracerProvider once. Safe to call more than once --
    only the first call takes effect.

    Without calling this, get_tracer() returns OpenTelemetry's own no-op
    tracer: spans are created but go nowhere, at essentially zero cost. This
    is what keeps graph.py's node/tool-call instrumentation unconditional --
    the test suite never calls configure_tracing(), so tracing is inert
    there rather than something tests need to set up or mock.
    """
    global _configured
    if _configured:
        return
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter or ConsoleSpanExporter()))
    trace.set_tracer_provider(provider)
    _configured = True


def get_tracer() -> Tracer:
    return trace.get_tracer(TRACER_NAME)


def traced_node(
    name: str, node_fn: Callable[[AgentState], dict[str, Any]]
) -> Callable[[AgentState], dict[str, Any]]:
    """Wrap a graph node function in a span named after the node."""

    def wrapped(state: AgentState) -> dict[str, Any]:
        with get_tracer().start_as_current_span(f"node.{name}") as span:
            span.set_attribute("guarded_agent.node", name)
            span.set_attribute("guarded_agent.budget.steps_used", state.budget.steps_used)
            span.set_attribute("guarded_agent.budget.tokens_used", state.budget.tokens_used)
            result = node_fn(state)
            budget = result.get("budget")
            if budget is not None:
                span.set_attribute("guarded_agent.budget.tokens_used_after", budget.tokens_used)
                span.set_attribute(
                    "guarded_agent.budget.tool_calls_used_after", budget.tool_calls_used
                )
            return result

    return wrapped


@contextmanager
def traced_tool_call(tool_name: str, arguments: dict[str, Any]) -> Iterator[Span]:
    """Span for a single tool dispatch, nested inside the executor node's span."""
    with get_tracer().start_as_current_span(f"tool.{tool_name}") as span:
        span.set_attribute("guarded_agent.tool.name", tool_name)
        span.set_attribute("guarded_agent.tool.arguments", json.dumps(arguments, default=str))
        yield span
