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

- **The ablation table above is empty on purpose.** Isolating each guardrail's
  contribution (`baseline` → `+registry` → `+policy_checker` → `+critic` → `full`)
  needs a real run across all five variants, and — since this is a personal project,
  not a submission with a budget behind it — that run hasn't been executed at
  meaningful scale. What's real today is a qualitative v1-vs-v2 comparison in
  [`docs/RESULTS.md`](docs/RESULTS.md): with the full guardrail stack on, Pass^1 drops
  sharply, but reading the actual transcripts shows most of that drop is the critic
  correctly catching a hallucination and safely handing off, not the agent getting
  worse — exactly the kind of nuance a single aggregate number hides.
- **The adversarial suite and failure-labeler calibration are built, not run at scale.**
  `evals/ablations.yaml`, `evals/run_suite.py`, and `analysis/mast_labeler.py` are all
  committed, tested, and working end-to-end — they just haven't been pointed at a real
  budget yet. Judge calibration (Cohen's kappa against hand labels) is currently
  measured against 6 failed traces, not the 40-60 a statistically meaningful number
  would need.

See [`docs/EVALUATION.md`](docs/EVALUATION.md#known-limitations--how-to-extend) for
exactly how to run the full versions of these later, including free-tier model options
that get this close to $0.

## License

MIT — see [LICENSE](LICENSE).
