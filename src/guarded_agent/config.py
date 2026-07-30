from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

ENV_PREFIX = "GUARDED_AGENT_"

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


class BudgetConfig(BaseModel):
    max_steps: int
    max_tool_calls: int
    max_tokens: int
    max_wall_clock_seconds: int


class RunConfig(BaseModel):
    domain: str
    agent_llm: str
    user_llm: str
    temperature: float
    num_trials: int
    num_tasks: int
    max_steps: int
    max_errors: int
    max_concurrency: int
    retry_delay: float
    max_retries: int
    save_to: str
    budget: BudgetConfig


def _apply_env_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> dict[str, Any]:
    """Overlay `GUARDED_AGENT_<FIELD>` env vars onto a loaded YAML dict.

    Only overrides top-level scalar fields; `budget.*` is left to YAML, since
    nothing outside this module needs to override it per-run yet.
    """
    merged = dict(raw)
    for field in RunConfig.model_fields:
        if field == "budget":
            continue
        env_key = f"{ENV_PREFIX}{field.upper()}"
        if env_key in env:
            merged[field] = env[env_key]
    return merged


def load_config(
    yaml_path: Path = DEFAULT_CONFIG_PATH, env: Mapping[str, str] | None = None
) -> RunConfig:
    """Load a `RunConfig` from a YAML file, with env-var overrides taking precedence."""
    raw = yaml.safe_load(yaml_path.read_text())
    merged = _apply_env_overrides(raw, os.environ if env is None else env)
    return RunConfig.model_validate(merged)
