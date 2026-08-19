# Evaluation

A reader should be able to reproduce every number in `RESULTS.md` from this file alone.
Everything below is verified against either the vendored `tau2` source directly (not
assumed from its docs) or this project's own already-committed code.

## What Pass^k measures, and why it isn't accuracy

Every task gets run `num_trials` times. For one task, Pass^k is the probability that if
you only got to sample k of those trials at random, *all k* would succeed:

```
pass_hat_k(num_trials, success_count, k) = C(success_count, k) / C(num_trials, k)
```

(`tau2.metrics.agent_metrics.pass_hat_k`, wrapped — never reimplemented — by this
project's `evals/run_suite.py`'s `pass_hat_k_table` and unit-tested against the
vendored function directly in `tests/test_run_suite.py`.) The headline Pass^k for a
whole run is the mean of each task's Pass^k across all tasks
(`tau2.metrics.agent_metrics.get_tasks_pass_hat_k`'s own aggregation, mirrored by
`pass_hat_k_table`).

Pass^1 *is* the ordinary success rate (mean reward across trials) — `C(s,1)/C(n,1) =
s/n`. The reason Pass^k for k>1 exists at all: **accuracy alone can't distinguish a
reliably-half-right agent from an inconsistent one.** An agent that succeeds on exactly
half of every task's trials (2/4, always) has the same Pass^1 as one that succeeds 4/4
on half its tasks and 0/4 on the other half — but Pass^2 (and especially Pass^4) for the
first agent stays close to its Pass^1, while the second agent's Pass^4 craters toward 0,
because "all 4 succeed" becomes vanishingly unlikely for a task where success is really
coin-flip-random per trial. Pass^k is a consistency probe hiding behind what looks like
a single number — this project's `RESULTS.md` rows so far are all single-trial
(`num_trials=1`), so only Pass^1 is reported; the ablation study (PLAN.md commit 24,
parked for budget reasons — see below) is the piece designed to also report Pass^4.

`is_successful` (`tau2.metrics.agent_metrics.is_successful`) is `1-1e-6 <= reward <=
1+1e-6` — a float-tolerant check for reward exactly 1.0, not "reward above some
threshold." Reward itself (`tau2.evaluator.evaluator.evaluate_simulation`, read
directly from the vendored source, not assumed) is a **product**, not an average, of
whichever sub-rewards a task's `reward_basis` includes — DB (database end-state match),
ACTION (individual tool-call correctness), NL_ASSERTION (LLM-judged natural-language
assertions about what was communicated), COMMUNICATE. A single 0.0 anywhere in that
product zeroes the whole task's reward, which is why this project's committed traces
repeatedly show `DB: 1.0, NL_ASSERTION: 0.0 → reward: 0.0`: the database action was
correct but a required communicated fact wasn't, and multiplication means partial
credit doesn't exist. A simulation that doesn't terminate via `AGENT_STOP`/`USER_STOP`
(hits `max_steps`, times out, errors) never reaches evaluation at all —
`reward=0.0, reward_basis=None` is assigned directly, no criteria checked.

## Dataset construction

Tasks come from τ²-bench's own committed task sets (`vendor/tau2-bench/data/domains/
<domain>/tasks.json`), never authored by this project — retail has 114 tasks total.
Every "N tasks" run in this project's committed results (`baseline_smoke`, `v1_smoke`,
`v2_smoke`: 5 each; the adversarial suite's 28 hand-authored policy-pressure cases are
the one exception, since those are testing pressure scenarios, not sampling the task
distribution) used `num_tasks=N` un-seeded — verified by reading
`tau2/runner/helpers.py`'s `get_tasks` directly: `tasks[:num_tasks]`, a **stable,
deterministic first-N slice of the task file's own order**, not a random or seeded
sample. This matters for reproducibility: re-running any of this project's 5-task
commands against the same τ²-bench version gets the *exact same* 5 tasks, and it's why
`evals/ablations.yaml`'s proposed 40-task run doesn't need explicit `task_ids` to stay
consistent across all 5 variants — `num_tasks=40` alone already guarantees that.

## Metric definitions (as computed by this project's own code)

- **Pass^k**: above.
- **Avg steps** / **Avg tokens** / **Cost / task**: `analysis.trace_loader.Trace`'s
  `num_steps` (message count), `total_tokens` (sum of `usage.prompt_tokens +
  usage.completion_tokens` across steps with usage), `total_cost` (`agent_cost +
  user_cost`, tau2's own per-simulation cost fields) — all per-conversation means
  across the committed run's simulations, computed directly by whatever script produced
  each `RESULTS.md` row (never hand-typed; CLAUDE.md rule 6).
- **Policy-violation rate** (adversarial suite only): `evals/adversarial/runner.py`'s
  `summarize()` — failed cases / total cases, where "failed" means one of three checks
  (`must_not_propose_action`, `must_not_leak_canary`, `must_gate_mutating_action`)
  returned false. `docs/RESULTS.md`'s "Adversarial suite" section documents a real
  correction to this number after a check-design bug was found and fixed — the
  committed raw JSON is unmodified; the corrected reading is explained in prose, not
  silently baked into the file.
- **MAST failure labels**: `analysis/mast_labeler.py`'s four deterministic checks
  (`schema_violation`, `budget_breach`, `non_termination`, `missing_confirmation`) run
  first and are authoritative when any fire; the LLM judge (`make_llm_judge_fn`) only
  judges the residual — a trace none of the four deterministic checks explain.

## Judge calibration (methodology; not yet run at scale)

PLAN.md commit 28's design: hand-label each failed trace with one of MAST's 14 failure
modes (see `docs/FAILURE_TAXONOMY.md`), independently get the LLM judge's label for the
same trace, then report **accuracy** (fraction where hand label == judge label) and
**Cohen's kappa** (agreement corrected for the rate expected by chance alone — the
standard reason to prefer kappa over raw accuracy for a 14-way categorical label,
since raw agreement can look artificially high if one or two categories dominate).
Originally scoped against 40-60 failed traces; only 6 exist in committed results today
(2 from `v1_smoke`, 4 from `v2_smoke`) since the ablation run that would have produced
many more (PLAN.md commit 24) is parked for the same budget reason as below. Whenever
this runs, the number gets published as-is, including if it's mediocre — CLAUDE.md rule
6 applies here too, not just to Pass^k.

## Known Limitations & How to Extend

This is a personal project, evaluated on a personal budget — not a submission with
sponsor credits behind it. Three pieces of the evaluation plan are fully built and
tested but deliberately not run at the scale originally planned, because doing so
costs real money:

| What | Built? | Run at scale? |
|---|---|---|
| Ablation study (`evals/ablations.yaml`, `evals/run_suite.py`) | Yes — 5 variants, verified via `--dry-run` and structural graph-node inspection, zero live cost | No — original plan was 5 variants × 40 tasks × 4 trials (~800 simulations, estimated $40-120) |
| Adversarial suite (`evals/adversarial/`) | Yes — 28 cases, run twice (free locally, then ~$0.24 against the production model). See [`RESULTS.md`'s "Adversarial suite"](RESULTS.md) for the full breakdown, including a real check-design bug found and fixed along the way | Done at the scale that fit the remaining budget — not a large-sample study, but a real one |
| Failure-labeler judge calibration (`analysis/mast_labeler.py`) | Yes — deterministic checks unit-tested, LLM judge fixture-tested | Only against 6 failed traces (from `v1_smoke` + `v2_smoke`), not the 40-60 needed for a statistically meaningful Cohen's kappa |

What's real instead: a qualitative v1-vs-v2 comparison in
[`RESULTS.md`](RESULTS.md), built by reading actual transcripts rather than trusting
one aggregate number — see that file for why Pass^1 dropping sharply from v1 to v2
mostly reflects the critic correctly catching mistakes and escalating safely, not the
agent getting worse.

### How to run the full versions later

No code changes needed — `llm_agent`/`llm_user` (in `configs/default.yaml`,
`evals/ablations.yaml`, and `evals/run_suite.py`'s `VariantConfig`) are plain strings
passed straight to `litellm`, which resolves the provider from the string prefix.
Running the ablation study as originally scoped is:

```bash
uv run --env-file .env python -m evals.run_suite --config evals/ablations.yaml
```

**To get close to $0**, swap `llm_agent`/`llm_user` in `evals/ablations.yaml` to a
free-tier provider instead of `anthropic/claude-haiku-4-5-20251001`. Verified directly
against the vendored `litellm` package's own model list, not assumed:

| Provider | Model string | Free tier (verify current numbers before relying on it — these change often) | API key env var |
|---|---|---|---|
| Ollama (fully local) | `ollama/<model>` (e.g. `ollama/llama3.1:latest`) | **The only one of these four that actually worked, live-verified 2026-08-18.** Unlimited, genuinely $0. Two distinct caveats found the hard way, both root-caused with a free local debug run rather than just asserted: (1) output *quality* is below Haiku — in the adversarial suite it defaulted to blanket escalation (`transfer_to_human_agents`) on 27 of 28 cases, including routine requests that should've gotten a plain text answer; (2) *protocol reliability* is the harder problem for anything needing tau2's real multi-turn simulation loop (the ablation study, not the adversarial suite, which only needs single scripted actions) — a debug run at `max_steps=12` showed the model calling `transfer_to_human_agents` four turns in a row without ever sending retail policy's required follow-up line ("YOU ARE BEING TRANSFERRED..."), so the conversation never actually terminates; it also once emitted a tool call as literal JSON text instead of a real structured tool call, and the user-simulator role leaked its own internal formatting markup ("### User:\n...") into a simulated customer message. Tool-calling isn't *broken*, it's *inconsistent* — and a full simulation loop needs many consecutive correct steps in a row, so one miss anywhere causes a runaway to the step cap. Fine for structural/single-action testing (adversarial suite); not currently viable for a full multi-turn ablation run. | none (runs locally) |
| Groq | `groq/llama-3.3-70b-versatile` | **Sign-up itself is broken (2026-08-18)**: new accounts (tried both email and GitHub OAuth) hit "does not belong to any organizations" — a known, reported bug in Groq's new-account provisioning where the auth layer succeeds but the account never gets attached to an organization. Never got as far as testing the actual API. | `GROQ_API_KEY` |
| Cerebras | `cerebras/gpt-oss-120b` or `cerebras/gemma-4-31b` | **Verified live (2026-08-15), not just documented:** advertised as "no card required," but this account's actual live model list (`GET /v1/models`) only exposed `gpt-oss-120b`/`gemma-4-31b` (not the `llama-3.3-70b` generally cited), and *both* returned `Payment required to access this resource` on a real call despite no card on file. Whether that's an account-specific quirk or the current real policy wasn't resolved — don't assume "no card required" without checking your own account's billing tab first. | `CEREBRAS_API_KEY` |
| Google AI Studio (Gemini) | `gemini/gemini-2.5-flash` | Already tried during this project (commit 5 era): actual observed cap was **20 requests/day on a fresh project**, not the ~1,500/day generally advertised. Not viable for a real run without a multi-day drip. | `GEMINI_API_KEY` |

The pattern across every one of these: the generally-advertised free-tier terms didn't
match this project's actual live experience in some way (a broken signup flow, an
unexpected billing wall, a tighter cap, or — for the one that did work — quality too low
to trust for anything where response quality itself is what's being measured). Verify
against your own account directly (a single minimal `litellm.completion(..., max_tokens=5)`
call costs a fraction of a cent and tells you immediately) before planning a real run
around any of them — and when the study needs to measure whether the *production* model
resists something (like the adversarial suite), a small amount of real spend against the
real model may be unavoidable and is usually cheap: the Haiku adversarial run above cost
$0.24 for 28 cases.

**The one cost a free agent/user model can't remove:** τ²-bench's own NL-assertion and
environment-interface evaluation calls are hardcoded to `gpt-4.1-2025-04-14`
(`tau2/config.py`'s `DEFAULT_LLM_NL_ASSERTIONS` / `DEFAULT_LLM_ENV_INTERFACE`, confirmed
by reading the vendored source — there is no CLI or config override) for every domain
that has NL assertions, which is every real domain this project uses (retail, airline,
telecom). A fully free run still needs *some* real `OPENAI_API_KEY` balance for that
piece specifically — much smaller than the full conversation cost, since it's judging
finished outcomes rather than carrying the dialogue, but not zero. The `mock` domain has
no NL assertions and is the one domain where a Groq/Cerebras/Ollama swap reaches
genuinely $0 end-to-end.
