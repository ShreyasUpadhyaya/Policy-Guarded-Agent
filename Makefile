.PHONY: install check test lint typecheck smoke-mock smoke-guarded eval-full report

install:
	uv sync --group dev

check: lint typecheck test

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy

smoke-mock:
	@echo "not available until PLAN.md commit 3 (pin and wire tau2-bench)" && exit 1

smoke-guarded:
	@echo "not available until PLAN.md commit 11 (tau2 adapter)" && exit 1

eval-full:
	@echo "not available until PLAN.md commit 24 (FULL ablation run) -- ASK BEFORE RUNNING" && exit 1

report:
	@echo "not available until PLAN.md commit 22 (eval harness) / analysis/report.py" && exit 1
