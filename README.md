# Policy-Guarded Agent

Policy-Guarded Agent is a multi-agent, policy-enforcing customer-service agent built on
LangGraph and evaluated against Sierra's τ²-bench. The deliverable isn't just a working
agent — it's evidence about its reliability: an ablation study isolating which guardrail
actually moves the pass rate, an adversarial suite measuring how often policy holds up
under pressure, and a MAST-grounded failure taxonomy built from real traces rather than
assumption.

## Results

| Variant | Pass^1 | Pass^4 | Policy violations | Avg steps | Cost / task |
|---|---|---|---|---|---|
| Off-the-shelf ReAct baseline | | | | | |
| + schema-validated tool registry | | | | | |
| + policy checker and write gate | | | | | |
| + critic with bounded retries | | | | | |
| Full system | | | | | |
| Full system, stronger model | | | | | |

## Quickstart

_Coming soon._

## Architecture

_Coming soon._

## Evaluation

_Coming soon._

## What still breaks

_Coming soon._

## License

MIT — see [LICENSE](LICENSE).
