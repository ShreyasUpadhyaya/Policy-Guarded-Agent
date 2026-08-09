from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]
    requestor: str


class TokenUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class Step(BaseModel):
    """One normalized message (assistant, user, or tool) from a simulation."""

    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None
    turn_idx: int
    cost: float | None = None
    usage: TokenUsage | None = None


class ActionCheck(BaseModel):
    """One evaluated tool call from reward_info.action_checks -- tau2's own
    ground truth for whether an action matched the task's expected action
    and whether it was a read or a write, independent of anything our own
    registry/graph believed at the time."""

    name: str
    tool_type: str
    action_match: bool


class Trace(BaseModel):
    """A single tau2 simulation, normalized from a results.json entry."""

    id: str
    task_id: str
    trial: int
    domain: str
    termination_reason: str
    agent_cost: float
    user_cost: float
    reward: float
    reward_breakdown: dict[str, float]
    steps: list[Step]
    action_checks: list[ActionCheck] = []

    @property
    def total_cost(self) -> float:
        return self.agent_cost + self.user_cost

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_tokens(self) -> int:
        return sum(
            step.usage.prompt_tokens + step.usage.completion_tokens
            for step in self.steps
            if step.usage is not None
        )


def _parse_step(raw: dict[str, Any]) -> Step:
    raw_tool_calls = raw.get("tool_calls")
    raw_usage = raw.get("usage")
    return Step(
        role=raw["role"],
        content=raw.get("content"),
        tool_calls=(
            [ToolCall.model_validate(tc) for tc in raw_tool_calls] if raw_tool_calls else None
        ),
        turn_idx=raw["turn_idx"],
        cost=raw.get("cost"),
        usage=TokenUsage.model_validate(raw_usage) if raw_usage else None,
    )


def _parse_action_check(raw: dict[str, Any]) -> ActionCheck:
    return ActionCheck(
        name=raw["action"]["name"], tool_type=raw["tool_type"], action_match=raw["action_match"]
    )


def _parse_trace(raw: dict[str, Any], domain: str) -> Trace:
    reward_info = raw["reward_info"]
    return Trace(
        id=raw["id"],
        task_id=raw["task_id"],
        trial=raw["trial"],
        domain=domain,
        termination_reason=raw["termination_reason"],
        agent_cost=raw["agent_cost"],
        user_cost=raw["user_cost"],
        reward=reward_info["reward"],
        reward_breakdown=reward_info.get("reward_breakdown", {}),
        steps=[_parse_step(m) for m in raw["messages"]],
        action_checks=[_parse_action_check(a) for a in reward_info.get("action_checks") or []],
    )


def load_traces(path: Path) -> list[Trace]:
    """Parse a tau2 `results.json` file into a list of normalized Traces."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    domain: str = raw["info"]["environment_info"]["domain_name"]
    return [_parse_trace(sim, domain) for sim in raw["simulations"]]
