# Google Gemini API Integration

This document outlines how Google's Gemini API integrates into the Policy-Guarded Agent pipeline and the scope of its usage.

## Overview

The Policy-Guarded Agent supports multiple LLM providers through litellm's abstraction layer. Google Gemini API can be used as an alternative to Anthropic Claude or OpenAI GPT models for agent reasoning and user simulation.

## Architecture Integration

### Pipeline Components

The agent pipeline has the following LLM-dependent stages:

```
Agent Input
    ↓
[1] Agent LLM (reasoning, tool decisions)
    ↓
[2] Policy Checker (guardrail validation)
    ↓
[3] Critic (response quality assurance)
    ↓
[4] τ²-bench Judge (outcome grading) ← hardcoded gpt-4.1
    ↓
Evaluation Results
```

### Gemini API Scope

#### ✅ **Can Use Gemini:**

1. **Agent LLM** (Stage 1)
   - Reasoning, tool-calling decisions, response generation
   - Config: `llm_agent: "gemini/gemini-3.0-flash"`
   - Use case: Primary agent reasoning
   - Cost: Free-tier or low-cost via Gemini API

2. **User Simulator** (τ²-bench internal)
   - Simulates user responses in conversations
   - Config: `llm_user: "gemini/gemini-3.0-flash"`
   - Use case: Multi-turn conversation simulation
   - Cost: Free-tier or low-cost via Gemini API

3. **Policy Checker** (Stage 2)
   - Inherits from agent LLM configuration
   - Validates proposed actions against policy
   - Automatically uses Gemini if agent is Gemini

4. **Critic** (Stage 3)
   - Inherits from agent LLM configuration
   - Validates response quality
   - Automatically uses Gemini if agent is Gemini

#### ❌ **Cannot Use Gemini:**

1. **τ²-bench Judge** (Stage 4) — HARDCODED OpenAI
   - Grades task success via NL-assertions
   - Implementation: `tau2/config.py` hardcodes `gpt-4.1-2025-04-14`
   - No CLI or config override available
   - **Cost**: Unavoidable OpenAI expense (~$0.005-0.01 per task)
   - **Reason**: Judge needs high reliability; τ²-bench team chose gpt-4.1 as the standard

## Configuration

### YAML Config Example

```yaml
# ablations.yaml or any run config
variants:
  - name: gemini-test
    agent: guarded_agent
    domain: retail
    llm_agent: gemini/gemini-3.0-flash
    llm_user: gemini/gemini-3.0-flash
    num_tasks: 5
    num_trials: 1
```

### Environment Setup

```bash
# .env file
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Still needed for judge
```

### Available Gemini Models

- **`gemini/gemini-3.0-flash`** (Recommended)
  - Latest, fastest, free-tier available
  - Best for both agent and user simulation
  
- **`gemini/gemini-2.0-flash`** (Alternative)
  - Previous generation, also free-tier
  - Fallback if 3.0 is unavailable

## Cost Analysis

### Single Task Run with Gemini Agent + OpenAI Judge

| Component | Model | Cost | Notes |
|-----------|-------|------|-------|
| Agent conversation | Gemini 3.0-flash | ~$0.00 | Free-tier quota or low-cost |
| Judge grading | gpt-4.1 (hardcoded) | ~$0.008 | Unavoidable OpenAI cost |
| **Total per task** | - | ~$0.008 | Judge dominates cost |

### Comparison with Pure OpenAI

| Variant | Agent Cost | Judge Cost | Total |
|---------|-----------|-----------|-------|
| Gemini + OpenAI Judge | ~$0.00 | ~$0.008 | ~$0.008 |
| OpenAI gpt-4.1-mini | ~$0.012 | ~$0.008 | ~$0.020 |
| Anthropic Haiku | ~$0.009 | ~$0.008 | ~$0.017 |

**Conclusion**: Gemini reduces agent cost to near-zero, but judge remains the bottleneck. Net savings: ~40-50% vs pure OpenAI.

## Integration Points in Code

### 1. Adapter (`src/guarded_agent/adapters/tau2_agent.py`)

The tau2 adapter translates between:
- Gemini API (via litellm) ↔ τ²-bench Message format
- Tool calls, streaming responses, cost tracking

**Key method**: `_generate()` at line ~190 calls `tau2_generate()` which routes through litellm to Gemini.

### 2. Eval Harness (`evals/run_suite.py`)

Accepts `llm_agent` and `llm_user` as configuration fields. No changes needed — litellm routing is transparent.

### 3. Guardrails (`src/guarded_agent/guardrails/`)

- `policy_checker.py`: Uses whatever LLM is passed in config
- `critic.py`: Same as above
- Both call `litellm.completion()` which routes to Gemini if configured

## Testing Gemini Integration

### Smoke Test (Single Task, 1 Trial)

```bash
# Create a quick test config
cat > evals/test_gemini.yaml << 'EOF'
variants:
  - name: gemini-smoke
    agent: guarded_agent
    domain: retail
    llm_agent: gemini/gemini-3.0-flash
    llm_user: gemini/gemini-3.0-flash
    num_tasks: 1
    num_trials: 1
EOF

# Run it
uv run python -m evals.run_suite --config evals/test_gemini.yaml
```

### Full Ablation with Gemini Agent (5 variants × 5 tasks × 1 trial)

```bash
# See: ablations_gemini.yaml (to be created)
uv run python -m evals.run_suite --config evals/ablations_gemini.yaml
```

## Known Limitations

1. **Judge is hardcoded to OpenAI**: No way to swap the grading LLM without modifying τ²-bench internals
2. **Gemini free-tier quotas**: Rate limits may apply (check Google AI Studio console)
3. **Cost tracking**: Judge cost is extracted from τ²-bench logs; agent cost comes from litellm responses

## Future Work

- Monitor Gemini API cost and performance vs other providers
- If τ²-bench ever exposes judge LLM as configurable, add Gemini option
- Consider Gemini for local development/testing (free-tier) and OpenAI for production runs

## References

- Google Gemini API: https://ai.google.dev/
- litellm Gemini provider: https://docs.litellm.ai/docs/providers/gemini
- τ²-bench hardcoded judge: `vendor/tau2-bench/tau2/config.py`
