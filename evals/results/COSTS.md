# Evaluation Run Costs

Tracks real spend for every run whose results are committed under `evals/results/`.
Numbers here are read directly from the committed `results.json` for that run, never
estimated. Exploratory/plumbing runs (e.g. the commit 3/4 mock-domain smoke checks used
only to verify tau2-bench wiring, never committed anywhere) aren't eval runs in this
sense and aren't logged here.

| Date | Run | Domain | Tasks x Trials | Agent LLM | User LLM | Total cost | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-29 | baseline_smoke | retail | 5 x 1 | anthropic/claude-haiku-4-5-20251001 | anthropic/claude-haiku-4-5-20251001 | $0.5103 | Off-the-shelf tau2 `llm_agent` baseline (PLAN.md commit 5). Also required a real `OPENAI_API_KEY`: tau2's NL-assertion and env-interface evaluation calls are hardcoded to `gpt-4.1-2025-04-14` with no CLI override, independent of --agent-llm/--user-llm. Pass^1 = 0.600 (3/5 tasks passed on genuine task criteria, not infra errors). |
