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

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

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
