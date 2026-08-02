from __future__ import annotations

from typing import Any

import pytest

from guarded_agent.tools.registry import ToolRegistry


@pytest.fixture
def registry() -> ToolRegistry:
    return ToolRegistry.load()


def test_registry_loads_expected_tools(registry: ToolRegistry) -> None:
    get_order = registry.get("get_order_details")
    refund = registry.get("issue_refund")

    assert get_order is not None
    assert get_order.mutating is False
    assert get_order.risk_tier == "low"

    assert refund is not None
    assert refund.mutating is True
    assert refund.risk_tier == "high"


def test_valid_call_dispatches_to_handler(registry: ToolRegistry) -> None:
    calls: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> dict[str, str]:
        calls.append(args)
        return {"status": "shipped"}

    result = registry.dispatch("get_order_details", {"order_id": "123"}, handler)

    assert result.ok is True
    assert result.result == {"status": "shipped"}
    assert calls == [{"order_id": "123"}]


def test_malformed_arguments_rejected_without_calling_handler(registry: ToolRegistry) -> None:
    calls: list[dict[str, Any]] = []

    def handler(args: dict[str, Any]) -> None:
        calls.append(args)
        raise AssertionError("handler should never be called for invalid arguments")

    # missing required "order_id"
    result = registry.dispatch("get_order_details", {}, handler)

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "invalid_arguments"
    assert calls == []


def test_wrong_argument_type_rejected_without_calling_handler(registry: ToolRegistry) -> None:
    def handler(args: dict[str, Any]) -> None:
        raise AssertionError("handler should never be called for invalid arguments")

    result = registry.dispatch("issue_refund", {"order_id": "123", "amount": "a lot"}, handler)

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "invalid_arguments"


def test_unexpected_extra_property_rejected(registry: ToolRegistry) -> None:
    def handler(args: dict[str, Any]) -> None:
        raise AssertionError("handler should never be called for invalid arguments")

    result = registry.dispatch(
        "get_order_details", {"order_id": "123", "unexpected": "field"}, handler
    )

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "invalid_arguments"


def test_unknown_tool_rejected_without_calling_handler(registry: ToolRegistry) -> None:
    def handler(args: dict[str, Any]) -> None:
        raise AssertionError("handler should never be called for an unknown tool")

    result = registry.dispatch("delete_everything", {}, handler)

    assert result.ok is False
    assert result.error is not None
    assert result.error.error_type == "unknown_tool"


SAMPLE_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Look up an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "cancel_pending_order",
            "description": "Cancel an order.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]


def test_from_openai_schemas_uses_real_mutating_status_when_given() -> None:
    registry = ToolRegistry.from_openai_schemas(
        SAMPLE_SCHEMAS,
        mutating_by_name={"get_order_details": False, "cancel_pending_order": True},
    )

    read_tool = registry.get("get_order_details")
    write_tool = registry.get("cancel_pending_order")

    assert read_tool is not None
    assert read_tool.mutating is False
    assert write_tool is not None
    assert write_tool.mutating is True


def test_from_openai_schemas_falls_back_to_default_mutating_for_unclassified_tools() -> None:
    registry = ToolRegistry.from_openai_schemas(
        SAMPLE_SCHEMAS, mutating_by_name={"get_order_details": False}
    )

    unclassified = registry.get("cancel_pending_order")

    assert unclassified is not None
    assert unclassified.mutating is True  # default_mutating=True: fail-safe forcing


def test_from_openai_schemas_defaults_everything_mutating_without_classification() -> None:
    registry = ToolRegistry.from_openai_schemas(SAMPLE_SCHEMAS)

    assert registry.get("get_order_details").mutating is True  # type: ignore[union-attr]
    assert registry.get("cancel_pending_order").mutating is True  # type: ignore[union-attr]
