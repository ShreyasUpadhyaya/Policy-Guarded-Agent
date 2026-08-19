# Policy-Guarded Agent

Policy-Guarded Agent is a policy-enforcing customer-service agent built on LangGraph and
evaluated against Sierra's τ²-bench. The deliverable isn't just a working agent — it's
evidence about its reliability: real transcripts read and quoted (not just an aggregate
score), a 28-case adversarial suite that found and helped fix real bugs in its own
guardrails, and a MAST-grounded failure taxonomy built from actual traces.

This is a personal project on a personal budget, not a funded submission — see
[What still breaks](#what-still-breaks) for exactly what that limited and how.

## Results

| Variant | Domain | Model (agent = user) | Tasks | Pass^1 | Avg steps | Avg tokens | Cost / task |
|---|---|---|---|---|---|---|---|
| Off-the-shelf τ²-bench `llm_agent` (baseline) | retail | claude-haiku-4-5 | 5 | 0.600 | 26.0 | 92,823 | $0.1021 |
| `guarded_agent` v1 — LangGraph, budgets + kill switch only | retail | claude-haiku-4-5 | 5 | 0.600 | 27.6 | 109,636 | $0.1190 |
| `guarded_agent` v2 — full guardrail stack (policy checker, write gate, escalation, critic) | retail | claude-haiku-4-5 | 5 | 0.000 | 24.0 | 68,985 | $0.0748 |

The v2 row isn't "the agent got worse." Reading the actual transcripts: 4 of 5 tasks
safely transferred to a human after the critic caught a real, verifiable mistake (an
unsupported product-stock claim, a policy misreading) on both its drafted response and
its one allowed revision — the guardrails traded raw task completion for catching
errors before they reached the customer. Full breakdown, including one case that looks
like the critic being *too* strict, in [`docs/RESULTS.md`](docs/RESULTS.md).

**Adversarial suite** (28 policy-pressure cases, [`docs/RESULTS.md`](docs/RESULTS.md#adversarial-suite)):
run against a free local model first (uninformative — it just escalated almost
everything), then against the production model for $0.24. Found and fixed two real bugs
in the test harness itself along the way, then landed on exactly one genuine
vulnerability: a fabricated prior-conversation-turn injected into a message got the
agent to skip real authentication and look up a real customer's order data.

**Ablation study** (isolating which specific guardrail moves the numbers above) is
built and tested but not run at meaningful scale — see
[What still breaks](#what-still-breaks).

## Quickstart

```bash
git clone https://github.com/ShreyasUpadhyaya/Policy-Guarded-Agent
cd Policy-Guarded-Agent
make install
cp .env.example .env   # fill in ANTHROPIC_API_KEY at minimum
make check             # lint + typecheck + full test suite -- zero API cost
```

`make check` is the guaranteed-to-work step: 218 tests, all fixture-based, no live API
calls (CLAUDE.md's own testing convention — every LLM-dependent behavior in this
codebase is tested via mocked/recorded responses, never a real call). To see the agent
actually run against a real model, `make smoke-mock` (2 tasks, the cheapest possible
domain) or `make smoke-guarded` (5 retail tasks) — both need your own funded API key.

## Architecture

LangGraph `StateGraph` where every node is a (near-)pure function of `AgentState` in,
state-update dict out: `agent` proposes an action or a reply; a proposed tool call goes
through `policy_gate` (retrieval-backed policy check) and `write_gate` (explicit
confirmation for mutating actions); a text reply goes through a `critic` with one
bounded revision; four independent triggers (budget breach, repeated tool failure,
policy deadlock, a second critic rejection) can escalate to a human at any point. Full
diagram, node-by-node responsibilities, the state schema, and a verified pure-vs-impure
module breakdown in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Evaluation

What Pass^k actually measures (a consistency probe across repeated trials, not just a
fancier average — and why that's different from accuracy), how reward is computed
(a *product* of applicable sub-criteria, not an average — one failing criterion zeroes
the whole task), dataset construction, and every metric in this README traced back to
the exact code that computes it: [`docs/EVALUATION.md`](docs/EVALUATION.md).

## What still breaks

Five real, open items — categorized honestly rather than smoothed over:

1. **The ablation study hasn't run at meaningful scale.** `evals/ablations.yaml` and
   `evals/run_suite.py` are built, tested, and verified structurally (each of the 5
   variants compiles the exact graph nodes it should — confirmed via live introspection,
   not just code review) — but isolating *which specific guardrail* drives the Pass^1
   drop in the Results table above needs a real run this project's remaining budget
   can't cover. ([`docs/EVALUATION.md`](docs/EVALUATION.md#known-limitations--how-to-extend)
   has exact commands and free-tier options for whoever runs this next.)
2. **Judge calibration is measured against 6 failed traces, not 40-60.** Cohen's kappa
   between hand labels and the LLM judge (`analysis/mast_labeler.py`) needs a larger
   sample than what's currently committed to mean much statistically — direct
   consequence of #1 (the ablation run would have produced many more failed traces to
   label).
3. **4 of the adversarial suite's 28 cases are structurally inconclusive, not passing.**
   Those 4 combine a real identity claim with the actual attack in one scripted message;
   the agent's correct, mandatory first move (authenticate) ends the turn before the
   attack itself is ever tested. Documented as inconclusive rather than silently counted
   as a pass — [`docs/RESULTS.md`](docs/RESULTS.md#adversarial-suite) has the detail.
4. **Free-tier models can't currently substitute for a full evaluation run.** Verified
   live, not assumed: a local Ollama model is fine for the adversarial suite's single
   scripted actions, but fails tau2's multi-step transfer protocol (calls
   `transfer_to_human_agents` repeatedly without ever sending the required follow-up
   line), causing full multi-turn simulations to loop to the step cap without ever
   being graded. Getting a real evaluation-quality result currently requires the
   production model, at real (if usually small) cost.
5. **The critic's strictness isn't independently calibrated.** One case in the v2 run
   looks like the critic rejecting a redundant-but-harmless re-confirmation as if it
   were a real error — plausible, not confirmed, and exactly the kind of question the
   parked ablation study (#1) exists to answer with more than one anecdote.

## τ²-bench submission

Not submitted to τ²-bench's leaderboard — this is a personal project built to learn and
demonstrate the guardrail architecture and evaluation methodology, not a competitive
benchmark entry. Everything needed to run the standard evaluation is present
(`make smoke-guarded`, `evals/run_suite.py`) if that changes later.

## License

MIT — see [LICENSE](LICENSE).
