from __future__ import annotations

from pathlib import Path

from analysis.trace_loader import load_traces

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "simulation_results.json"


def test_load_traces_returns_one_trace_per_simulation() -> None:
    traces = load_traces(FIXTURE_PATH)
    assert len(traces) == 2


def test_domain_is_read_from_file_level_info() -> None:
    traces = load_traces(FIXTURE_PATH)
    assert all(t.domain == "retail" for t in traces)


def test_trace_fields_match_source_json() -> None:
    traces = load_traces(FIXTURE_PATH)
    first = traces[0]

    assert first.id == "sim-1"
    assert first.task_id == "0"
    assert first.termination_reason == "user_stop"
    assert first.reward == 1.0
    assert first.reward_breakdown == {"DB": 1.0, "NL_ASSERTION": 1.0}
    assert first.total_cost == first.agent_cost + first.user_cost


def test_num_steps_counts_every_message_including_tool() -> None:
    traces = load_traces(FIXTURE_PATH)
    assert traces[0].num_steps == 4
    assert traces[1].num_steps == 2


def test_tool_calls_parsed_from_assistant_message() -> None:
    traces = load_traces(FIXTURE_PATH)
    tool_call_step = traces[0].steps[2]

    assert tool_call_step.tool_calls is not None
    assert len(tool_call_step.tool_calls) == 1
    assert tool_call_step.tool_calls[0].name == "find_user_id_by_name_zip"
    assert tool_call_step.tool_calls[0].arguments["last_name"] == "Rossi"


def test_message_without_tool_calls_or_usage_defaults_to_none() -> None:
    traces = load_traces(FIXTURE_PATH)
    greeting = traces[0].steps[0]

    assert greeting.tool_calls is None
    assert greeting.usage is None


def test_tool_role_message_parses_without_cost_or_usage_fields() -> None:
    traces = load_traces(FIXTURE_PATH)
    tool_result = traces[0].steps[3]

    assert tool_result.role == "tool"
    assert tool_result.content == "yusuf_rossi_9620"
    assert tool_result.cost is None
    assert tool_result.usage is None


def test_total_tokens_sums_usage_across_steps() -> None:
    traces = load_traces(FIXTURE_PATH)
    # sim-1: (400+12) + (5650+106) = 6168; the greeting and tool-result steps have no usage.
    assert traces[0].total_tokens == 6168
