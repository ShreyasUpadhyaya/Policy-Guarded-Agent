# AGENTS.md

Instructions for Codex working in this repository. Read this before every task.

## What this project is

A multi-agent, policy-enforcing customer-service agent built on LangGraph and evaluated on Sierra's τ²-bench. The deliverable is not just a working agent: it is **evidence about the agent's reliability**. Evaluation code, failure analysis, and documentation are first-class, not afterthoughts.

The full commit-by-commit plan is in `PLAN.md`. Work through it in order.

## Hard rules

1. **Never run a FULL benchmark evaluation without asking me first.** FULL runs cost real money. SMOKE runs (5 tasks, 1 trial, cheap model) are fine without asking. If a command would exceed the smoke tier, stop and ask.
2. **Never commit secrets.** `.env` is gitignored. If you need a new key, add it to `.env.example` with a placeholder value.
3. **One commit per task.** Do not batch multiple `PLAN.md` rows into one commit. Do not touch files outside the current task's scope.
4. **`make check` must pass before every commit.** If it fails, fix it or stop and report; do not commit anyway and do not weaken a test to make it pass.
5. **Do not modify vendored τ²-bench code.** Extend it through the adapter in `src/guarded_agent/adapters/`. If you think you need to patch the benchmark, stop and ask. Where the vendored package already implements something we need (e.g. `tau2.metrics.agent_metrics.pass_hat_k`), wrap it — do not reimplement it.
6. **Do not invent numbers.** Every figure in `RESULTS.md`, `README.md`, or `docs/` must come from a committed results file under `evals/results/`. If a number is not measured yet, leave the cell empty.
7. **Redact before committing traces.** Trace samples in `traces/` must have keys, emails, and simulated PII stripped.
8. **τ²-bench is never a plain PyPI dependency.** It installs as import name `tau2`, but PyPI's `tau2` is an unrelated package (magnetic relaxation rate calculations). `tau2-bench` does not exist on PyPI either. Always add it as a **git source** pinned to an exact tag (`[tool.uv.sources]` in `pyproject.toml`), never `uv add tau2`.

## Commands

```bash
make install       # uv sync, install dev extras
make check         # lint + typecheck + tests. Run before every commit
make test          # pytest only
make lint          # ruff check + ruff format --check
make smoke-mock    # 2 tasks on the mock domain, cheapest possible sanity run
make smoke-guarded # 5 retail tasks x 1 trial with our agent (SMOKE tier, allowed)
make eval-full     # FULL tier. ASK BEFORE RUNNING
make report        # regenerate RESULTS.md tables from evals/results/
```

## Code conventions

- Python 3.12+ (pinned to `>=3.12,<3.14` — the range required by the vendored `tau2` package itself), `uv` for dependency management, `ruff` for lint and format, `pytest` for tests.
- Type hints on every public function. Pydantic models for anything crossing a boundary (state, tool args, config, trace records).
- No bare `except`. Tool failures become structured error objects, never silent passes.
- Every LangGraph node is a pure-ish function of state: it reads state, returns a state update. Side effects go through the tool registry so they can be logged and gated.
- `guardrails/`, the tool registry's schema validation, and escalation-trigger logic are held to a harder standard: zero I/O. No network call, no clock read, no LLM call. They are pure functions of state, which is what makes them unit-testable without mocks and reproducible byte-for-byte.
- Prompts live in `src/guarded_agent/prompts/` as separate files, never inline in logic. They are versioned and diffable.
- Model names, temperatures, and budgets come from config, never hardcoded.

## Testing conventions

- **Deterministic logic gets real unit tests**: the tool registry validator, budget enforcement, Pass^k computation, trace parsing, escalation triggers, memory retention filters. These must not call an LLM.
- **LLM-dependent behaviour gets fixture-based tests**: record a response once, replay it. No live API calls in the test suite.
- Write the test in the same commit as the feature, not later.
- When a bug is found, add the failing test first, then fix.

## Repository layout

```
src/guarded_agent/
  config.py            # env + YAML config
  state.py             # typed agent state
  graph.py             # LangGraph assembly
  nodes/               # router, planner, policy_checker, executor, critic, escalation
  tools/               # registry.yaml + schema-validating dispatcher
  guardrails/          # budgets, write gate, kill switch
  memory/              # session state, case store, retention policy
  adapters/            # tau2-bench integration
  telemetry/           # tracing and span emission
  prompts/             # versioned prompt files
evals/
  run_suite.py         # eval harness, computes Pass^k
  ablations.yaml       # variant definitions
  adversarial/         # injection and policy-pressure cases
  results/             # committed run outputs + COSTS.md
analysis/
  trace_loader.py      # tau2 JSON -> typed Trace
  mast_labeler.py      # deterministic checks + LLM judge for failure classification
  report.py            # generates markdown tables and charts
docs/                  # ARCHITECTURE, EVALUATION, FAILURE_TAXONOMY, MEMORY, RESULTS
traces/                # redacted sample traces for reviewers
tests/
```

## How to start a task

1. Read the relevant row in `PLAN.md`, including its acceptance check.
2. Restate the task and the acceptance check back to me in one or two sentences.
3. Propose a file-level plan. List exactly which files you will create or modify. If the list extends beyond the task's scope, cut it.
4. Wait for approval, then implement.
5. Run `make check`. Show me the diff summary and the test output.
6. Propose a conventional-commit message whose body names the acceptance check that passed.

## Things that have gone wrong before (avoid these)

- Adding retry logic without a bound, producing agents that loop until the budget cap fires. Every loop needs an explicit maximum and a test that proves it.
- Writing a "policy check" that just asks the model whether the action is allowed, with no retrieved clause and no cited clause id. The verdict must reference a specific policy clause.
- Letting the critic rewrite the answer into vagueness to avoid criticism. The critic returns a verdict and a reason, and only one revision is permitted.
- Silently swallowing tool schema violations. A violation is a recorded event with a failure label, not a retry that hides the problem.
- Letting a degraded or low-confidence input (retrieval failure, tool error, missing context) resolve to a permissive default. Degraded inputs must force the conservative branch — `NEEDS_CONFIRMATION` or escalate — never a silent `ALLOW`.
- Updating README numbers by hand. Numbers are generated by `make report` from committed results.
