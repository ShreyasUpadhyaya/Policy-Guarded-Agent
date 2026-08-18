# Evaluation

Full methodology (what Pass^k measures, dataset construction, metric definitions,
judge calibration) is `PLAN.md` commit 29 — not written yet. This file currently has
one section, written early because it directly explains real, committed decisions:
why the ablation/adversarial/calibration studies aren't run at full scale yet, and
exactly how to run them later without necessarily spending anything close to the
original estimate.

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
| Ollama (fully local) | `ollama/<model>` (e.g. `ollama/llama3.1:latest`) | **The only one of these four that actually worked, live-verified 2026-08-18.** Unlimited, genuinely $0. Caveat found the hard way: output quality/calibration is meaningfully below Haiku — in the adversarial suite it defaulted to blanket escalation (`transfer_to_human_agents`) on 27 of 28 cases, including routine requests that should've gotten a plain text answer, which made its results useless as a robustness signal even though free. Fine for structural testing; not a substitute for the production model when the *quality* of the response is what's being measured. | none (runs locally) |
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
