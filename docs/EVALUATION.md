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
| Adversarial suite (`evals/adversarial/`) | Not yet (PLAN.md commit 25) | N/A |
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
| Groq | `groq/llama-3.3-70b-versatile` | ~30 requests/min, 6K tokens/min, no card required | `GROQ_API_KEY` |
| Cerebras | `cerebras/llama-3.3-70b` | ~1M tokens/day, no card required — better fit than Groq if a guarded-agent conversation's per-turn token count (policy checker + critic calls add real overhead) exceeds Groq's per-minute cap | `CEREBRAS_API_KEY` |
| Ollama (fully local) | `ollama/<model>` (e.g. `ollama/llama3.1:8b`) | Unlimited, genuinely $0 — bounded only by local RAM/CPU, and output quality is meaningfully below Haiku | none (runs locally) |
| Google AI Studio (Gemini) | `gemini/gemini-2.5-flash` | Already tried during this project (commit 5 era): actual observed cap was **20 requests/day on a fresh project**, not the ~1,500/day generally advertised. Not viable for a real run without a multi-day drip. |

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
