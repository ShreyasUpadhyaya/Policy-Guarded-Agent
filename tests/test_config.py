from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from guarded_agent.config import DEFAULT_CONFIG_PATH, load_config

MINIMAL_YAML = """
domain: mock
agent_llm: anthropic/claude-haiku-4-5-20251001
user_llm: anthropic/claude-haiku-4-5-20251001
temperature: 0.0
num_trials: 1
num_tasks: 2
max_steps: 200
max_errors: 10
max_concurrency: 3
retry_delay: 1.0
max_retries: 4
save_to: wrapper-smoke

budget:
  max_steps: 20
  max_tool_calls: 20
  max_tokens: 50000
  max_wall_clock_seconds: 300

retrieval:
  top_k: 3
  min_confidence: 0.5
"""


def test_committed_default_config_loads() -> None:
    config = load_config(DEFAULT_CONFIG_PATH, env={})
    assert config.domain == "mock"
    assert config.budget.max_tool_calls > 0


def test_load_config_from_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(MINIMAL_YAML)

    config = load_config(yaml_path, env={})

    assert config.num_tasks == 2
    assert config.agent_llm == "anthropic/claude-haiku-4-5-20251001"
    assert config.budget.max_tokens == 50000


def test_env_override_wins_over_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(MINIMAL_YAML)

    config = load_config(yaml_path, env={"GUARDED_AGENT_NUM_TASKS": "5"})

    assert config.num_tasks == 5


def test_invalid_value_raises_validation_error(tmp_path: Path) -> None:
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text(MINIMAL_YAML.replace("num_trials: 1", "num_trials: not-a-number"))

    with pytest.raises(ValidationError):
        load_config(yaml_path, env={})
