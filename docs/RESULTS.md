# Results

| Variant | Domain | Model (agent = user) | Tasks | Trials | Pass^1 | Avg steps | Avg tokens | Cost / task |
|---|---|---|---|---|---|---|---|---|
| Off-the-shelf tau2 `llm_agent` (baseline) | retail | anthropic/claude-haiku-4-5-20251001 | 5 | 1 | 0.600 | 26.0 | 92,823 | $0.1021 |
| Our `guarded_agent` v1 (LangGraph, no guardrails) | retail | anthropic/claude-haiku-4-5-20251001 | 5 | 1 | 0.600 | 27.6 | 109,636 | $0.1190 |

Baseline (PLAN.md commit 5): off-the-shelf tau2 `llm_agent` — no guardrails, no custom
architecture, the number the rest of the project has to beat.

Agent v1 (PLAN.md commit 14): our own LangGraph agent, running inside τ²-bench via the
tau2 adapter (commit 11), with execution budgets and a kill switch (commit 12) but no
policy checker, write gate, or critic yet — those start Day 3. Pass^1 matches the
baseline exactly (0.600), and both hit DB Match 5/5 (100%): both agents fail the exact
same two tasks (task 2 and task 4), and both times purely on the NL-assertion criterion
(`DB: 1.0, NL_ASSERTION: 0.0`) rather than the underlying database action. Avg tokens
and cost/task are both higher for v1 (~18%), plausibly the cost of a longer, more
cautious system prompt with no guardrails yet to make it more efficient. That tradeoff
is exactly what the ablation study (Day 4) exists to measure precisely instead of
eyeballing.

Sources: [`evals/results/baseline_smoke/results.json`](../evals/results/baseline_smoke/results.json),
[`evals/results/v1_smoke/results.json`](../evals/results/v1_smoke/results.json)
(both committed), cost ledger in [`evals/results/COSTS.md`](../evals/results/COSTS.md).
Computed with `analysis.trace_loader.load_traces` (commit 6): avg steps and avg tokens
are per-conversation means; Pass^1 is the fraction of tasks with reward 1.0.
