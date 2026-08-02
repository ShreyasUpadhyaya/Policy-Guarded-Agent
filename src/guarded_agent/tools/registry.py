from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel

DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parent / "registry.yaml"


class ToolDefinition(BaseModel):
    name: str
    description: str
    mutating: bool
    risk_tier: Literal["low", "medium", "high"]
    parameters: dict[str, Any]


class ToolError(BaseModel):
    error_type: Literal["unknown_tool", "invalid_arguments"]
    message: str
    details: list[str] = []


class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    result: Any = None
    error: ToolError | None = None


class ToolRegistry:
    def __init__(self, tools: dict[str, ToolDefinition]) -> None:
        self._tools = tools

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> ToolRegistry:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        tools = {t["name"]: ToolDefinition.model_validate(t) for t in raw["tools"]}
        return cls(tools)

    @classmethod
    def from_openai_schemas(
        cls,
        schemas: list[dict[str, Any]],
        *,
        mutating_by_name: dict[str, bool] | None = None,
        default_mutating: bool = True,
        risk_tier: Literal["low", "medium", "high"] = "high",
    ) -> ToolRegistry:
        """Build a registry from OpenAI-style function-calling schemas.

        Used for domains (e.g. a tau2 benchmark run) whose real tools aren't
        known ahead of time and so can't live in registry.yaml. `schemas`
        alone carry no read/write classification -- pass `mutating_by_name`
        when the caller has real per-tool classification available (e.g. the
        tau2 adapter reads it off each Tool's underlying function). Any name
        not in `mutating_by_name` falls back to `default_mutating` (True by
        default: fail-safe forcing, per CLAUDE.md, rather than assuming a
        tool not otherwise classified is safe).
        """
        mutating_by_name = mutating_by_name or {}
        tools = {}
        for schema in schemas:
            function = schema["function"]
            name = function["name"]
            tools[name] = ToolDefinition(
                name=name,
                description=function.get("description", ""),
                mutating=mutating_by_name.get(name, default_mutating),
                risk_tier=risk_tier,
                parameters=function.get("parameters", {"type": "object", "properties": {}}),
            )
        return cls(tools)

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    @property
    def tools(self) -> dict[str, ToolDefinition]:
        return dict(self._tools)

    def dispatch(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        handler: Callable[[dict[str, Any]], Any],
    ) -> ToolResult:
        """Validate `arguments` against the tool's schema, then call `handler`.

        Never raises on bad input: an unknown tool or a schema violation is
        returned as a structured ToolResult(ok=False, error=...) and `handler`
        is never invoked.
        """
        definition = self._tools.get(tool_name)
        if definition is None:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=ToolError(
                    error_type="unknown_tool",
                    message=f"No tool registered with name '{tool_name}'.",
                ),
            )

        validator = Draft202012Validator(definition.parameters)
        violations = sorted(validator.iter_errors(arguments), key=lambda e: e.path)
        if violations:
            return ToolResult(
                ok=False,
                tool_name=tool_name,
                error=ToolError(
                    error_type="invalid_arguments",
                    message=f"Arguments for '{tool_name}' failed schema validation.",
                    details=[v.message for v in violations],
                ),
            )

        return ToolResult(ok=True, tool_name=tool_name, result=handler(arguments))
