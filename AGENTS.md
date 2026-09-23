# Agent Variants for Ablation Studies

This file documents the different agent configurations used for ablation studies to measure the impact of each guardrail stage.

## Variants Overview

The Policy-Guarded Agent is designed with cumulative guardrail stages:

1. **baseline** — No guardrails, pure LangGraph agent with budgets only
2. **+registry** — Add tool schema validation and registry
3. **+policy_checker** — Add policy-based action validation
4. **+critic** — Add response quality review with bounded revision
5. **full** — All guardrails enabled (same as `guarded_agent` default)

## Agent Factory Registration

All variants are registered in [src/guarded_agent/adapters/tau2_agent.py](src/guarded_agent/adapters/tau2_agent.py):

```python
ABLATION_AGENT_VARIANTS = {
    "guarded_agent_baseline": make_guarded_agent_baseline,
    "guarded_agent_registry": make_guarded_agent_registry,
    "guarded_agent_policy_checker": make_guarded_agent_policy_checker,
    "guarded_agent_critic": make_guarded_agent_critic,
    "guarded_agent": make_guarded_agent,  # full, default
}
```

## Running Ablation Studies

### Full Ablation (Original Plan)
```bash
uv run python -m evals.run_suite --config evals/ablations.yaml
```
- 5 variants × 40 tasks × 4 trials
- Requires explicit budget approval per CLAUDE.md rule 1
- Estimated cost: $40-120 on Anthropic/OpenAI

### Smoke Test (Quick Validation)
```bash
make smoke-guarded
```
- 5 variants × 5 tasks × 1 trial
- Minimal cost, good for structure validation
- Built-in make target

### Gemini Integration Test
```bash
uv run python -m evals.run_suite --config evals/ablations_gemini.yaml
```
- 5 variants × 5 tasks × 1 trial using Gemini 3.0-flash
- Agent/user cost: free-tier (Gemini)
- Judge cost: OpenAI only (~$0.04 total)
- See [docs/GEMINI_INTEGRATION.md](docs/GEMINI_INTEGRATION.md) for details

### Small OpenAI Slice
```bash
uv run python -m evals.run_suite --config evals/ablations_small_openai.yaml
```
- 5 variants × 5 tasks × 1 trial using gpt-4.1-mini
- Cost-efficient real-world baseline
- Used for Day 4-5 evaluation work

## Cost Breakdown by Variant

| Config | Models | Cost/Task | Tasks | Trials | Total |
|--------|--------|-----------|-------|--------|-------|
| ablations.yaml | Haiku | ~$0.017 | 40 | 4 | ~$2.7k |
| ablations_small_openai.yaml | gpt-4.1-mini | ~$0.020 | 5 | 1 | ~$0.10 |
| ablations_gemini.yaml | Gemini 3.0 | ~$0.008 | 5 | 1 | ~$0.04 |
| smoke-guarded (make) | Haiku | ~$0.017 | 5 | 1 | ~$0.09 |

## Guardrail Impact Hypothesis

Expected Pass^1 progression across variants:

- **baseline**: Highest raw task completion (no guardrails to block actions)
- **+registry**: Slight drop (schema validation catches invalid calls)
- **+policy_checker**: Larger drop (policy limits some legitimate actions)
- **+critic**: Similar or slightly lower (critic catches hallucinations)
- **full**: Stable but conservative (all guardrails active)

Trade-off: raw task completion vs. safety and accuracy of completed actions.

## Provider Flexibility

All variants support any litellm-compatible LLM provider via config:

```yaml
variants:
  - name: variant-name
    llm_agent: gemini/gemini-3.0-flash  # Google
    llm_user: gemini/gemini-3.0-flash
```

```yaml
    llm_agent: anthropic/claude-haiku-4-5-20251001  # Anthropic
    llm_user: anthropic/claude-haiku-4-5-20251001
```

```yaml
    llm_agent: gpt-4.1-mini  # OpenAI
    llm_user: gpt-4.1-mini
```

## See Also

- [docs/GEMINI_INTEGRATION.md](docs/GEMINI_INTEGRATION.md) — Detailed Gemini API integration guide
- [docs/EVALUATION.md](docs/EVALUATION.md) — Full evaluation methodology and limitations
- [PLAN.md](PLAN.md) — Commit-by-commit development plan including ablation design
