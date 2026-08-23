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

**Ablation study** (isolating which specific guardrail moves the numbers above), small
but real — 5 variants × 5 tasks, run twice against increasingly capable OpenAI models
after six free-tier providers (Groq, Cerebras, Gemini, Cohere, two local Qwen/Llama
models) were each ruled out for a distinct, live-verified reason
([`docs/RESULTS.md`](docs/RESULTS.md#ablation-study)). `gpt-4.1-nano` ($0.12) proved
too weak to pass the DB check on any variant; `gpt-4.1-mini` ($0.58) completed
cleanly and found baseline (no guardrails) outperforming every guardrail-enabled
variant — a real, if small-sample (n=5), signal that the critic/policy_checker may be
stricter than necessary. Not the original 40×4-task design — see
[What still breaks](#what-still-breaks) for the honest scope and caveats.

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

1. **The ablation study has run, but only small — n=5 tasks per variant, not the
   original 40×4.** `evals/ablations.yaml` and `evals/run_suite.py` are built, tested,
   and now also run for real: 5 variants × 5 tasks × 1 trial against `gpt-4.1-mini`
   ($0.58, after `gpt-4.1-nano` at $0.12 proved too weak to pass the DB check on any
   variant at all). The result is a real but not statistically meaningful data point —
   baseline outperformed every guardrail-enabled variant, which is a genuine, if
   small-sample, signal rather than proof. ([`docs/RESULTS.md`](docs/RESULTS.md#ablation-study)
   has the full provider journey, both real result files, and exact honest caveats;
   [`docs/EVALUATION.md`](docs/EVALUATION.md#known-limitations--how-to-extend) has
   the command to run it at the original scale if budget opens up.)
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
   live against six candidates, not assumed: Groq (broken signup), Cerebras (billing
   wall despite "no card required"), Google AI Studio (a real 20 req/day cap, not the
   ~1,500 advertised), Cohere (a litellm tool-schema incompatibility), and two local
   Ollama models each failing differently — `llama3.1` loops on tau2's multi-step
   transfer protocol without ever sending the required follow-up line, and `qwen3`/
   `qwen2.5` (tried specifically for their tool-calling reputation) either never issue
   a real tool call at all or fabricate a completed customer action that never
   happened. Getting a real evaluation-quality result currently requires a paid model,
   at real (if usually small) cost — full breakdown in
   [`docs/EVALUATION.md`](docs/EVALUATION.md#how-to-run-the-full-versions-later).
5. **The critic/policy_checker's strictness isn't independently calibrated — now with
   a real, small data point, not just an anecdote.** The ablation run above (#1) found
   baseline (no guardrails) passing more tasks than every guardrail-enabled variant,
   including full. At n=5/variant that's not proof, but it's a genuine directional
   signal consistent with the [Results table](docs/RESULTS.md)'s v2 case where the
   critic rejected a redundant-but-harmless re-confirmation as if it were a real
   error — now looking like a real pattern, not a one-off.

## τ²-bench submission

Not submitted to τ²-bench's leaderboard — this is a personal project built to learn and
demonstrate the guardrail architecture and evaluation methodology, not a competitive
benchmark entry. Everything needed to run the standard evaluation is present
(`make smoke-guarded`, `evals/run_suite.py`) if that changes later.

## License

MIT — see [LICENSE](LICENSE).
