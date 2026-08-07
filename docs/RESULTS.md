# Results

| Variant | Domain | Model (agent = user) | Tasks | Trials | Pass^1 | Avg steps | Avg tokens | Cost / task |
|---|---|---|---|---|---|---|---|---|
| Off-the-shelf tau2 `llm_agent` (baseline) | retail | anthropic/claude-haiku-4-5-20251001 | 5 | 1 | 0.600 | 26.0 | 92,823 | $0.1021 |
| Our `guarded_agent` v1 (LangGraph, no guardrails) | retail | anthropic/claude-haiku-4-5-20251001 | 5 | 1 | 0.600 | 27.6 | 109,636 | $0.1190 |
| Our `guarded_agent` v2 (+ policy checker, write gate, escalation, critic) | retail | anthropic/claude-haiku-4-5-20251001 | 5 | 1 | 0.000 | 24.0 | 68,985 | $0.0748 |

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

Agent v2 (PLAN.md commit 20): adds the full guardrail stack — retrieval-backed policy
checker (commit 16), write gate with explicit confirmation (commit 17), three-trigger
escalation (commit 18), and a critic with one bounded revision (commit 20). Pass^1 drops
to 0.000, driven almost entirely by one pattern: **4 of 5 tasks safely transferred to a
human** after the critic rejected a drafted response on both its first pass and its one
allowed revision. Reading the actual transcripts (not just the reward), every one of
those four rejections was catching something real — an unsupported claim about a
product's stock status not actually shown in the tool results, a policy misreading about
when to batch order modifications into one call, and (more debatably) a redundant
re-confirmation of something the user had already explicitly approved. The fifth task
completed its write action correctly (DB Match 1/1 for that task) but separately failed
an unrelated NL assertion (a general product-catalog question our policy checker
correctly denies — the retail policy doesn't authorize listing store-wide inventory to
an unauthenticated or even authenticated caller, only order- and profile-scoped lookups).
So the 0.000 here is not "the agent got worse at the task" in the v1 sense — every
failure is either a deliberate, policy-grounded refusal or a safe hand-off after catching
a real mistake before it reached the customer. Whether that trade (large drop in raw
task completion for a large gain in caught-before-sending errors) is the right one for a
production deployment — and whether the critic's bar for "unsupported claim" is
calibrated correctly (the redundant-confirmation case above suggests it may sometimes be
stricter than necessary) — is exactly what the ablation study (Day 4, PLAN.md commit 23)
is designed to isolate per guardrail, rather than eyeballing one combined number.

Four real bugs were found and fixed live while producing this row (not ablation
findings, correctness bugs): `agent_revise` escalating instead of proposing a tool call
on revision, rejected tool-call proposals left dangling in conversation history (breaking
the *next* turn's LLM call outright), a policy checker with no visibility into the
conversation (denying already-authenticated actions again), and escalated messages
reporting cost as unknown rather than zero (breaking this project's own trace loader).
See `PLAN.md`'s environment gotchas for detail and the commits fixing each.

Sources: [`evals/results/baseline_smoke/results.json`](../evals/results/baseline_smoke/results.json),
[`evals/results/v1_smoke/results.json`](../evals/results/v1_smoke/results.json),
[`evals/results/v2_smoke/results.json`](../evals/results/v2_smoke/results.json)
(all committed), cost ledger in [`evals/results/COSTS.md`](../evals/results/COSTS.md).
Computed with `analysis.trace_loader.load_traces` (commit 6) and
`tau2.metrics.agent_metrics.pass_hat_k`: avg steps and avg tokens are per-conversation
means; Pass^1 is the fraction of tasks with reward 1.0.
