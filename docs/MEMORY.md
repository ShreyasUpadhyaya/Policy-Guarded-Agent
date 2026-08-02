# Memory

Two kinds of memory exist in this project, at two different lifetimes.

## Session state (per-task, in-process only)

`AgentState` (`src/guarded_agent/state.py`) is the full session state:
the entire message history, budget counters, the last policy verdict, a
pending confirmation, and escalation status. It is threaded through every
turn of a single tau2 task by `GuardedTau2Agent` (`adapters/tau2_agent.py`)
-- `get_init_state` builds it, `generate_next_message` updates and returns
it each turn.

It lives only in process memory for the duration of one task. It is never
written to disk by the agent itself (tau2's own harness separately persists
full transcripts under `evals/results/` for evaluation purposes -- that is
tau2's concern, not this module's). It is discarded when the task ends.

**Never redacted, because never stored**: session state can carry raw PII
(a real order ID, a real email the simulated user typed) because it never
leaves process memory and is never written anywhere by `guarded_agent`
itself.

## Case store (cross-session, resolved-case summaries)

`src/guarded_agent/memory/` builds a small retrieval-backed store of
**redacted summaries** of finished sessions, meant to let a future task
retrieve how a similar issue was handled before. It reuses the same
LangChain vectorstore/embeddings pattern as the policy retriever
(`guardrails/policy_retrieval.py`, commit 15): `HuggingFaceEmbeddings` +
FAISS, not a bespoke index.

### What is stored

`memory/session.py`'s `CaseRecord`, built by `build_case_record(state,
session_id)` from a session's *final* `AgentState`:

| Field | Source | Notes |
|---|---|---|
| `session_id` | caller-supplied | opaque id, not a real customer identifier |
| `issue_summary` | first user message | passed through `redact_pii` |
| `resolution_summary` | last assistant text reply | passed through `redact_pii` |
| `resolved` | `not state.escalated` | |
| `escalation_reason` | `state.escalation_reason` | free text set by our own escalation node, not user input |
| `tool_names_used` | distinct tool names from the conversation | names only, never arguments |

### What is never stored

- Raw conversation history. Only the two derived summary fields above ever
  reach the case store, and both are redacted first.
- Tool call **arguments** or tool **results** -- these are the fields most
  likely to carry a real order ID, address, or payment detail. Only the
  tool *name* is kept.
- The policy verdict's cited clause id or reason text.
- Anything if `redact_pii` cannot run on it -- `build_case_record` always
  redacts before assigning to `CaseRecord`, so there is no code path that
  stores unredacted free text.

### Retention filter

`memory/retention.py`'s `redact_pii` is a pure, zero-I/O regex filter run
on both free-text fields before a `CaseRecord` is built. It replaces:

- card-like digit sequences (13-19 digits, optionally grouped by spaces or
  dashes) with `[REDACTED_CARD]`
- email addresses with `[REDACTED_EMAIL]`
- phone numbers with `[REDACTED_PHONE]`

This is a heuristic, not a guarantee -- it matches PII by *shape*, not by
field name, so it will not catch PII that doesn't match one of these
shapes (e.g. a plain name typed in prose). It is deliberately conservative
in the other direction: an order ID that happens to be a 13+ digit run
also gets redacted, which is an acceptable false positive for a case
summary that only needs to be roughly retrievable, not exact.

### Lifetime

The case store is in-process only, same as session state -- there is no
disk-backed persistence in this commit. A `CaseStore` instance holds cases
for as long as the process that created it runs (e.g. one eval suite
run). Nothing here writes case records across process restarts.
