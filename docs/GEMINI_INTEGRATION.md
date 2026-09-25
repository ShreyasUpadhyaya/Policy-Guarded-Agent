# Gemini integration: scope and status

This project does **not** use Google ADK or the Gemini SDK. There is no
`google-*`, `genai`, or `adk` dependency or import anywhere in `pyproject.toml` or
`src/`. Gemini is reachable only the way every other provider is: as a litellm model
string passed through config. This page documents where such a string can go, what
it cannot replace, and what happened when it was actually tried.

## Where a Gemini model can be plugged in

Every model is chosen per variant in the eval config (`llm_agent`, `llm_user`); no
code change is needed to switch provider.

| LLM call site | Configured by | Can be Gemini? |
|---|---|---|
| Agent (the LangGraph agent's reasoning and tool calls) | `llm_agent` | Yes |
| Policy checker | same model as `llm_agent` (`adapters/tau2_agent.py` passes the agent's `llm` to `make_llm_policy_check_fn`) | Yes, follows `llm_agent` |
| Critic | same model as `llm_agent` (passed to `make_llm_critic_check_fn`) | Yes, follows `llm_agent` |
| User simulator (τ²-bench) | `llm_user` | Yes |
| NL-assertion judge (τ²-bench) | hardcoded in vendored `tau2/config.py` as `DEFAULT_LLM_NL_ASSERTIONS = "gpt-4.1-2025-04-14"` | **No.** No config override; changing it would mean patching vendored code (CLAUDE.md rule 5). Every graded run needs a working `OPENAI_API_KEY`. |

## What happened when it was tried

The live findings are recorded in
[`EVALUATION.md`'s free-tier provider table](EVALUATION.md#how-to-run-the-full-versions-later):

- `gemini-2.5-flash`: now returns 404 ("no longer available to new users").
- `gemini-3.6-flash`: passed a plain-text check and a real tool-schema call, then
  failed a real bounded single-task run with `429 RESOURCE_EXHAUSTED`. The free-tier
  quota is 5 requests per minute per project per model. One multi-turn conversation
  (agent, user simulator, and guardrail calls) goes past that before the first task
  finishes.

No graded Gemini result exists, so there are no Gemini Pass^k or cost numbers in
this repo. Any such figure would be invented (CLAUDE.md rule 6).

## How to try it again

[`evals/ablations_gemini.yaml`](../evals/ablations_gemini.yaml) is a template: the
same five cumulative variants as `evals/ablations.yaml`, with `llm_agent` and
`llm_user` set to `gemini/gemini-3.6-flash`. It needs `GEMINI_API_KEY` (a Google AI
Studio key, not an `sk-...` key) and `OPENAI_API_KEY` for the judge. It has not been
run. On the free tier it will hit the quota above, so it is only useful with a paid
Gemini quota. Even then, it still spends OpenAI credit on every judge call.
