# Evaluation Run Costs

Tracks real spend for every run whose results are committed under `evals/results/`.
Numbers here are read directly from the committed `results.json` for that run, never
estimated. Exploratory/plumbing runs (e.g. the commit 3/4 mock-domain smoke checks used
only to verify tau2-bench wiring, never committed anywhere) aren't eval runs in this
sense and aren't logged here.

| Date | Run | Domain | Tasks x Trials | Agent LLM | User LLM | Total cost | Notes |
|---|---|---|---|---|---|---|---|
| 2026-07-29 | baseline_smoke | retail | 5 x 1 | anthropic/claude-haiku-4-5-20251001 | anthropic/claude-haiku-4-5-20251001 | $0.5103 | Off-the-shelf tau2 `llm_agent` baseline (PLAN.md commit 5). Also required a real `OPENAI_API_KEY`: tau2's NL-assertion and env-interface evaluation calls are hardcoded to `gpt-4.1-2025-04-14` with no CLI override, independent of --agent-llm/--user-llm. Pass^1 = 0.600 (3/5 tasks passed on genuine task criteria, not infra errors). |
| 2026-07-30 | v1_smoke | retail | 5 x 1 | anthropic/claude-haiku-4-5-20251001 | anthropic/claude-haiku-4-5-20251001 | $0.5948 | Our own `guarded_agent` (LangGraph, no guardrails yet -- policy checker/write gate/critic are Day 3), commit 14. Pass^1 = 0.600, matching the baseline exactly; DB Match 5/5 (100%, vs baseline's 4/5). Zero kill-switch activations (budget recalibrated in commit 12). |
| 2026-08-08 | v2_smoke | retail | 5 x 1 | anthropic/claude-haiku-4-5-20251001 | anthropic/claude-haiku-4-5-20251001 | $0.3741 | Full guardrail stack (policy checker, write gate, escalation, critic), commit 20. Pass^1 = 0.000 (4/5 tasks safely transferred to a human after the critic caught a genuine hallucination or policy misstep on both the first draft and its one bounded revision; 1/5 completed its write action correctly but failed an unrelated NL assertion). See `docs/RESULTS.md` for the full breakdown. This is the run actually committed; three earlier live attempts the same day surfaced and got fixed as separate commits (an escalation-on-tool-call bug, an orphaned-`tool_use`-block bug, a context-blind policy checker, and a cost-aggregation bug) -- those attempts (~$0.85 combined across the two whose totals were recoverable, plus unlogged partial-failure runs before that) aren't logged as their own rows since their results were never committed, per this file's own convention for exploratory/plumbing runs. |
