# Results

## Baseline (Day 1)

Off-the-shelf tau2 `llm_agent` — no guardrails, no custom architecture. This is the
number the rest of the project has to beat.

| Domain | Model (agent = user) | Tasks | Trials | Pass^1 | Avg steps | Avg tokens | Cost / task |
|---|---|---|---|---|---|---|---|
| retail | anthropic/claude-haiku-4-5-20251001 | 5 | 1 | 0.600 | 26.0 | 92,823 | $0.1021 |

Total run cost: $0.5103 (5 tasks x $0.1021 avg).

Source: [`evals/results/baseline_smoke/results.json`](../evals/results/baseline_smoke/results.json)
(committed, PLAN.md commit 5), cost ledger in
[`evals/results/COSTS.md`](../evals/results/COSTS.md). Computed with
`analysis.trace_loader.load_traces` (commit 6): avg steps and avg tokens are
per-conversation means across the 5 simulations; Pass^1 is the fraction of tasks
with reward 1.0.
