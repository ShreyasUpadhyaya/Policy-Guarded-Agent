from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tau2.data_model.simulation import RewardInfo, SimulationRun, TerminationReason
from tau2.metrics.agent_metrics import pass_hat_k

from evals.run_suite import (
    VariantConfig,
    load_variants,
    outcomes_by_task,
    pass_hat_k_table,
    run_variant,
    summarize,
)

VARIANTS_YAML = """
variants:
  - name: baseline
    llm_agent: anthropic/claude-haiku-4-5-20251001
    llm_user: anthropic/claude-haiku-4-5-20251001
    num_tasks: 5
  - name: full
    llm_agent: anthropic/claude-haiku-4-5-20251001
    llm_user: anthropic/claude-haiku-4-5-20251001
    num_tasks: 5
"""


def _sim(task_id: str, reward: float | None, agent_cost: float | None = 0.01) -> SimulationRun:
    reward_info = RewardInfo(reward=reward, reward_basis=[]) if reward is not None else None
    return SimulationRun(
        id=f"{task_id}-sim",
        task_id=task_id,
        timestamp="2026-01-01T00:00:00Z",
        start_time="2026-01-01T00:00:00Z",
        end_time="2026-01-01T00:00:01Z",
        duration=1.0,
        termination_reason=(
            TerminationReason.USER_STOP
            if reward is not None
            else TerminationReason.INFRASTRUCTURE_ERROR
        ),
        reward_info=reward_info,
        agent_cost=agent_cost,
        user_cost=0.005 if agent_cost is not None else None,
    )


# --- pass_hat_k_table: PLAN.md commit 22's literal acceptance check ---------


def test_pass_hat_k_table_matches_vendored_pass_hat_k_for_a_single_task() -> None:
    outcomes = {"task-1": [True, True, False, True]}

    table = pass_hat_k_table(outcomes)

    for k in range(1, 5):
        assert table[k] == pytest.approx(pass_hat_k(4, 3, k))


def test_pass_hat_k_table_averages_across_tasks() -> None:
    outcomes = {
        "task-1": [True, True],  # 2/2 successes
        "task-2": [True, False],  # 1/2 successes
    }

    table = pass_hat_k_table(outcomes)

    expected_k1 = (pass_hat_k(2, 2, 1) + pass_hat_k(2, 1, 1)) / 2
    expected_k2 = (pass_hat_k(2, 2, 2) + pass_hat_k(2, 1, 2)) / 2
    assert table[1] == pytest.approx(expected_k1)
    assert table[2] == pytest.approx(expected_k2)


def test_pass_hat_k_table_caps_k_at_the_smallest_task_trial_count() -> None:
    outcomes = {"task-1": [True, True, True], "task-2": [True, False]}

    table = pass_hat_k_table(outcomes)

    assert set(table.keys()) == {1, 2}


def test_pass_hat_k_table_empty_input_returns_empty_table() -> None:
    assert pass_hat_k_table({}) == {}


# --- outcomes_by_task --------------------------------------------------------


def test_outcomes_by_task_groups_success_and_failure() -> None:
    simulations = [_sim("t1", reward=1.0), _sim("t1", reward=0.0), _sim("t2", reward=1.0)]

    grouped = outcomes_by_task(simulations)

    assert grouped == {"t1": [True, False], "t2": [True]}


def test_outcomes_by_task_excludes_infra_errors() -> None:
    simulations = [_sim("t1", reward=1.0), _sim("t1", reward=None)]

    grouped = outcomes_by_task(simulations)

    assert grouped == {"t1": [True]}


# --- summarize / run_variant --------------------------------------------------


def test_summarize_reports_infra_error_count_and_avg_cost() -> None:
    simulations = [
        _sim("t1", reward=1.0, agent_cost=0.02),
        _sim("t1", reward=None, agent_cost=None),
    ]

    result = summarize("baseline", simulations)

    assert result.total_simulations == 2
    assert result.total_tasks == 1
    assert result.infra_error_count == 1
    assert result.avg_cost == pytest.approx(0.02 + 0.005)


def test_run_variant_never_calls_run_domain_with_wrong_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_variant should build a real TextRunConfig and hand it to
    tau2.runner.batch.run_domain -- verified via a stub instead of a live
    call, per CLAUDE.md's fixture-based testing convention."""
    captured = {}

    class _StubResults:
        simulations = [_sim("t1", reward=1.0)]

    def _stub_run_domain(config):
        captured["config"] = config
        return _StubResults()

    monkeypatch.setattr("evals.run_suite.run_domain", _stub_run_domain)

    variant = VariantConfig(
        name="full", llm_agent="fake-agent-model", llm_user="fake-user-model", num_tasks=5
    )
    result = run_variant(variant)

    assert captured["config"].llm_agent == "fake-agent-model"
    assert captured["config"].llm_user == "fake-user-model"
    assert captured["config"].num_tasks == 5
    assert captured["config"].auto_resume is True
    assert result.name == "full"
    assert result.total_tasks == 1


# --- load_variants / --dry-run -----------------------------------------------


def test_load_variants_parses_yaml(tmp_path: Path) -> None:
    yaml_path = tmp_path / "variants.yaml"
    yaml_path.write_text(VARIANTS_YAML)

    variants = load_variants(yaml_path)

    assert [v.name for v in variants] == ["baseline", "full"]
    assert all(v.num_tasks == 5 for v in variants)


def test_dry_run_lists_variants_without_calling_run_domain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    yaml_path = tmp_path / "variants.yaml"
    yaml_path.write_text(VARIANTS_YAML)

    def _never_run_domain(config):
        raise AssertionError("run_domain should never be called in --dry-run mode")

    monkeypatch.setattr("evals.run_suite.run_domain", _never_run_domain)
    monkeypatch.setattr(sys, "argv", ["run_suite.py", "--config", str(yaml_path), "--dry-run"])

    from evals.run_suite import main

    main()

    out = capsys.readouterr().out
    assert "baseline" in out
    assert "full" in out
