from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from tau2.data_model.simulation import SimulationRun, TextRunConfig
from tau2.metrics.agent_metrics import is_successful, pass_hat_k
from tau2.runner.batch import run_domain

from guarded_agent.adapters import tau2_agent  # noqa: F401  (registers guarded_agent on import)

DEFAULT_LLM_ARGS = {"temperature": 0.0}


class VariantConfig(BaseModel):
    """One row of a run_suite.py config list -- a single tau2 run to execute."""

    name: str
    domain: str = "retail"
    agent: str = "guarded_agent"
    llm_agent: str
    llm_user: str
    num_trials: int = 1
    num_tasks: int | None = None
    task_ids: list[str] | None = None
    seed: int = 300


class VariantResult(BaseModel):
    """Machine-readable summary for one variant's run."""

    name: str
    total_simulations: int
    total_tasks: int
    infra_error_count: int
    pass_hat_k: dict[int, float]
    avg_cost: float


def load_variants(path: Path) -> list[VariantConfig]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [VariantConfig.model_validate(v) for v in raw["variants"]]


def _to_tau2_config(variant: VariantConfig) -> TextRunConfig:
    return TextRunConfig(
        domain=variant.domain,
        agent=variant.agent,
        llm_agent=variant.llm_agent,
        llm_args_agent=DEFAULT_LLM_ARGS,
        user="user_simulator",
        llm_user=variant.llm_user,
        llm_args_user=DEFAULT_LLM_ARGS,
        num_trials=variant.num_trials,
        num_tasks=variant.num_tasks,
        task_ids=variant.task_ids,
        save_to=variant.name,
        seed=variant.seed,
        auto_resume=True,
    )


def pass_hat_k_table(outcomes_by_task: dict[str, list[bool]]) -> dict[int, float]:
    """Pass^k for k=1..(smallest per-task trial count), wrapping tau2's own
    pass_hat_k for the actual combinatorics (PLAN.md commit 22: do not
    reimplement it).

    Each task's Pass^k is computed individually, then averaged across tasks
    for a given k -- the same aggregation
    tau2.metrics.agent_metrics.get_tasks_pass_hat_k itself uses, just without
    requiring a full tau2 Results/Simulation object graph to exercise: this
    operates on a plain per-task pass/fail matrix, so it's unit-testable with
    synthetic data alone.
    """
    if not outcomes_by_task:
        return {}
    max_k = min(len(outcomes) for outcomes in outcomes_by_task.values())
    table: dict[int, float] = {}
    for k in range(1, max_k + 1):
        per_task = [
            pass_hat_k(len(outcomes), sum(outcomes), k) for outcomes in outcomes_by_task.values()
        ]
        table[k] = sum(per_task) / len(per_task)
    return table


def outcomes_by_task(simulations: list[SimulationRun]) -> dict[str, list[bool]]:
    """Group each simulation's success/failure by task_id, in run order.

    Excludes simulations with no reward_info (infrastructure errors that
    never got far enough to be judged on task criteria) -- the same
    exclusion tau2.metrics.agent_metrics.get_metrics_df applies before
    computing Pass^k.
    """
    grouped: dict[str, list[bool]] = {}
    for sim in simulations:
        if sim.reward_info is None:
            continue
        grouped.setdefault(sim.task_id, []).append(is_successful(sim.reward_info.reward))
    return grouped


def summarize(name: str, simulations: list[SimulationRun]) -> VariantResult:
    grouped = outcomes_by_task(simulations)
    infra_errors = len(simulations) - sum(len(v) for v in grouped.values())
    costs = [
        sim.agent_cost + (sim.user_cost or 0.0) for sim in simulations if sim.agent_cost is not None
    ]
    return VariantResult(
        name=name,
        total_simulations=len(simulations),
        total_tasks=len(grouped),
        infra_error_count=infra_errors,
        pass_hat_k=pass_hat_k_table(grouped),
        avg_cost=sum(costs) / len(costs) if costs else 0.0,
    )


def run_variant(variant: VariantConfig) -> VariantResult:
    config = _to_tau2_config(variant)
    results = run_domain(config)
    return summarize(variant.name, results.simulations)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a list of tau2 eval variants.")
    parser.add_argument(
        "--config", type=Path, required=True, help="YAML file with a variants list."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="List variant names without running them."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("evals/results/run_suite_output.json"),
        help="Where to write the machine-readable summary.",
    )
    args = parser.parse_args()

    variants = load_variants(args.config)

    if args.dry_run:
        for variant in variants:
            print(variant.name)
        return

    results_out: list[dict[str, Any]] = [run_variant(variant).model_dump() for variant in variants]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"timestamp": datetime.now(UTC).isoformat(), "variants": results_out},
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
