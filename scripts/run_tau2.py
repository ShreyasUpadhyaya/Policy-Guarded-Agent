from __future__ import annotations

import subprocess

from guarded_agent.config import RunConfig, load_config


def build_run_command(config: RunConfig) -> list[str]:
    return [
        "tau2",
        "run",
        "--domain",
        config.domain,
        "--agent-llm",
        config.agent_llm,
        "--user-llm",
        config.user_llm,
        "--num-trials",
        str(config.num_trials),
        "--num-tasks",
        str(config.num_tasks),
        "--max-steps",
        str(config.max_steps),
        "--max-errors",
        str(config.max_errors),
        "--max-concurrency",
        str(config.max_concurrency),
        "--retry-delay",
        str(config.retry_delay),
        "--max-retries",
        str(config.max_retries),
        "--save-to",
        config.save_to,
    ]


def main() -> int:
    config = load_config()
    subprocess.run(["tau2", "check-data"], check=True)
    subprocess.run(build_run_command(config), check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
