# Failure Taxonomy

This project classifies observed failures against **MAST** (Multi-Agent System
Failure Taxonomy), from Cemri et al., ["Why Do Multi-Agent LLM Systems
Fail?"](https://arxiv.org/abs/2503.13657) (NeurIPS 2025 Datasets & Benchmarks
Track). MAST defines 14 failure modes across 3 categories, built from 150
hand-annotated traces and validated at κ = 0.88 inter-annotator agreement.

## Adapting MAST to this system

MAST was built for frameworks where multiple LLM agents converse with each
other directly (planner, executor, critic, etc. as separate chat
participants). This project isn't that shape: from τ²-bench's point of view
there are two parties (`guarded_agent` and the simulated `user`), and our own
"multi-agent-like" structure lives *inside* `guarded_agent` as a LangGraph
pipeline of specialized nodes (`agent` → `policy_gate` → `write_gate` →
`critic` → `escalation`) rather than as separate chat participants.

We apply MAST's categories at both levels, and say explicitly which level
each example is drawn from:

- **agent ↔ user**: the two-party τ²-bench conversation. "Other agent" in
  MAST's inter-agent-misalignment definitions maps to "the user."
- **internal pipeline**: our own nodes acting as gatekeeping/verification
  roles for each other (the critic reviewing the agent's draft, the policy
  gate reviewing the agent's proposed action). A failure here is our own
  guardrail code misbehaving, not the base LLM.

Every example below is a real observed trace, cited by file + `task_id`, from
already-committed results — nothing here is synthetic or hypothetical.

## FC1 — System Design Issues

| Mode | Definition |
|---|---|
| FM-1.1 Disobey task specification | Failure to adhere to the specified constraints or requirements of a task. |
| FM-1.2 Disobey role specification | Failure to adhere to an assigned role's defined responsibilities/constraints. |
| FM-1.3 Step repetition | Unnecessary reiteration of previously completed steps. |
| FM-1.4 Loss of conversation history | Unexpected context truncation, reverting to an earlier conversational state. |
| FM-1.5 Unaware of termination conditions | Lack of recognition of the criteria that should end the interaction. |

**Observed: FM-1.1, agent ↔ user.** [`evals/results/v1_smoke/results.json`](../evals/results/v1_smoke/results.json),
task `2`. The task's expected outcome is that the agent tells the user there
are **10** t-shirt options currently available. v1 (no guardrails yet)
answered:

> "We currently have **12 different t-shirt variants** available in our
> store... Most variants are in stock, though a couple are..."

12 is the total catalog count; 10 is the count *currently available* — a
different, unasked-for question answered instead of the one the task
specified. `reward_breakdown: {"DB": 1.0, "NL_ASSERTION": 0.0}` — the
database action was fine, only this informational answer failed the task's
actual specification. This exact failure mode is why commit 16's policy
checker later denies this question outright rather than guessing at it (see
FC3 below) — trading a wrong-answer failure for a policy-consistent refusal.

## FC2 — Inter-Agent Misalignment

| Mode | Definition |
|---|---|
| FM-2.1 Conversation reset | Unwarranted restarting of a dialogue, losing context and progress. |
| FM-2.2 Fail to ask for clarification | Not requesting more information when facing unclear/incomplete data. |
| FM-2.3 Task derailment | Deviation from the intended objective or focus of the task. |
| FM-2.4 Information withholding | Failing to share data/insights that could affect another party's decisions. |
| FM-2.5 Ignored other agent's input | Disregarding or under-weighting input another party already provided. |
| FM-2.6 Reasoning-action mismatch | The action taken doesn't match the agent's own stated reasoning. |

**Observed: FM-2.5, agent ↔ user.** [`evals/results/v2_smoke/results.json`](../evals/results/v2_smoke/results.json),
task `3`. The user had already explicitly confirmed ("Go ahead and make that
change") a single, specific modification. The agent's drafted follow-up
re-asked whether there were other items to modify, disregarding that the
scope had already been settled. Caught by our own critic node before it
reached the user:

> "The draft asks the user to confirm whether there are other items to
> modify, but the user has already explicitly confirmed they want to
> proceed with only the t-shirt change... this additional confirmation
> request contradicts the user's clear instruction and delays action
> unnecessarily."

Notable: this is a case where the failure was *caught and corrected*, not
one that reached the user — included here because MAST's failure modes are
about what the base agent produced, independent of whether a guardrail
downstream intervened.

## FC3 — Task Verification

| Mode | Definition |
|---|---|
| FM-3.1 Premature termination | Ending the interaction before all necessary information/objectives are met. |
| FM-3.2 No or incomplete verification | (Partial) omission of checking task outcomes or outputs. |
| FM-3.3 Incorrect verification | Failure to adequately validate/cross-check crucial information or decisions. |

**Observed: FM-3.2, agent ↔ user (v1, no verification step at all).**
Same trace as the FC1 example above
([`v1_smoke/results.json`](../evals/results/v1_smoke/results.json), task
`2`): v1 had no critic and no policy checker yet (commit 14 — Day 3's
guardrails didn't exist), so nothing checked the "12 vs. 10" answer against
either the tool results or the policy before it was sent. The same trace
illustrates both FC1 (wrong answer to the specified question) and FC3 (zero
verification existed to catch it).

**Observed: FM-3.3, *internal pipeline* — our own critic making the call, not
the base agent.** [`evals/results/v2_smoke/results.json`](../evals/results/v2_smoke/results.json),
task `1`. The critic rejected a drafted claim that a keyboard variant was
"currently unavailable," reasoning:

> "...the tool results show this item exists with available=false, which is
> an unsupported assertion about current stock status that goes beyond what
> the data confirms."

Read literally, `available=false` is exactly what "currently unavailable"
means — the critic's own justification here is internally inconsistent, not
obviously a correct catch. We're including it specifically *because* it's
ambiguous: it's an honest example of our verification layer's own fallibility
(FM-3.3, applied to the critic itself), not just a curated case where the
critic looked good. `docs/EVALUATION.md`'s judge-calibration work (PLAN.md
commit 28) is exactly the mechanism that would need to run at scale to know
how often this happens.

## Domain-specific additions

MAST's 14 modes were built from general-purpose multi-agent frameworks, not
from a policy-guardrail pipeline specifically. Three real, already-fixed bugs
from this project's own history don't fit cleanly into any single MAST mode
above — they're about the *guardrail machinery* itself, not the base agent's
reasoning:

**DG-1: Guardrail context blindness.** A guardrail LLM call (policy checker)
evaluates a proposed action using only that action's arguments and retrieved
policy text, with no visibility into what already happened in the
conversation — so a prerequisite the policy states (e.g. "authenticate
first") can never be recognized as *already satisfied*, and the guardrail
denies the same already-authenticated action repeatedly. Fixed in
`fix: give the policy checker conversation context` (commit history, found
live producing [`v2_smoke`](../evals/results/v2_smoke/results.json)'s first
attempt, superseded by the committed version). Closest MAST analogue: FM-1.4
(loss of conversation history), but the mechanism is different — nothing was
lost, the data was simply never given to that component in the first place.

**DG-2: Escalation over-triggering on a legitimate recovery.** A bounded
revision step (the critic's one allowed retry) treated *any* tool call from
the regenerated response as an unusable revision and escalated immediately,
even when calling a tool was exactly the right recovery from a rejected,
ungrounded claim. Fixed in `fix: agent_revise should propose a tool call, not
escalate on one`. Closest MAST analogue: FM-3.1 (premature termination), but
triggered by our own pipeline's logic rather than the base agent choosing to
stop.

**DG-3: Rejected proposals left dangling in history.** When a guardrail
rejects a proposed tool call, the rejection must *replace* the proposal in
conversation history, not sit alongside it — an orphaned `tool_use` block
with no matching `tool_result` breaks the provider's API contract on the
*next* call outright (not a soft failure — a hard `400` from Anthropic).
Fixed in `fix: remove rejected tool-call proposals from conversation
history`. No MAST analogue — this is a mechanical conversation-integrity bug
specific to how tool-calling APIs pair proposals with results, not a
reasoning failure at all.

All three are documented in more detail, with the exact live symptom, in
`PLAN.md`'s environment gotchas (commit 21 section) and `docs/RESULTS.md`.
