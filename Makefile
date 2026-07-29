.PHONY: install check test lint typecheck smoke-mock smoke-guarded eval-full report

install:
	uv sync --group dev
	test -d vendor/tau2-bench || git clone --depth 1 --branch v1.0.1 https://github.com/sierra-research/tau2-bench vendor/tau2-bench

check: lint typecheck test

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

smoke-mock:
	uv run --env-file .env tau2 check-data
	uv run --env-file .env tau2 run --domain mock --agent-llm anthropic/claude-haiku-4-5-20251001 --user-llm anthropic/claude-haiku-4-5-20251001 --num-trials 1 --num-tasks 2 --save-to smoke-mock

smoke-guarded:
	@echo "not available until PLAN.md commit 11 (tau2 adapter)" && exit 1

eval-full:
	@echo "not available until PLAN.md commit 24 (FULL ablation run) -- ASK BEFORE RUNNING" && exit 1

report:
	@echo "not available until PLAN.md commit 22 (eval harness) / analysis/report.py" && exit 1
