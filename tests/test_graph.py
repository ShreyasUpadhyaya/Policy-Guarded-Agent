from __future__ import annotations

import json
from pathlib import Path

import pytest

from guarded_agent.config import BudgetConfig
from guarded_agent.graph import AgentDecision, GenerateFn, build_graph, router, run
from guarded_agent.guardrails.critic import CriticCheckFn, CriticVerdict
from guarded_agent.guardrails.policy_checker import PolicyCheckFn
from guarded_agent.guardrails.policy_retrieval import PolicyContext, RetrievedClause
from guarded_agent.state import (
    AgentState,
    BudgetCounters,
    Message,
    PolicyVerdict,
    ProposedAction,
    ToolCall,
)
from guarded_agent.tools.registry import ToolDefinition, ToolRegistry

MOCK_TASKS_PATH = (
    Path(__file__).resolve().parents[1]
    / "vendor"
    / "tau2-bench"
    / "data"
    / "tau2"
    / "domains"
    / "mock"
    / "tasks.json"
)


def _stub_text(content: str) -> GenerateFn:
    def _generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        return AgentDecision(content=content)

    return _generate


def _stub_tool_call(tool_call: ToolCall) -> GenerateFn:
    def _generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        return AgentDecision(tool_call=tool_call)

    return _generate


def _stub_text_sequence(contents: list[str]) -> GenerateFn:
    """Returns `contents[0]` on the first call, `contents[1]` on the second,
    and so on, holding on the last entry for any further call -- lets a test
    give distinct answers to the initial agent call vs. agent_revise's call."""
    calls = {"n": 0}

    def _generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        content = contents[min(calls["n"], len(contents) - 1)]
        calls["n"] += 1
        return AgentDecision(content=content)

    return _generate


def test_router_picks_agent_when_no_proposed_action() -> None:
    state = AgentState()
    assert router(state) == "agent"


def test_router_picks_executor_when_proposed_action_set() -> None:
    state = AgentState(
        proposed_action=ProposedAction(tool_name="get_order_details", arguments={"order_id": "1"})
    )
    assert router(state) == "executor"


def test_agent_only_path_appends_assistant_message() -> None:
    app = build_graph(_stub_text("Is there anything else I can help you with?"))
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert [m.role for m in result.conversation] == ["user", "assistant"]
    assert result.conversation[-1].content == "Is there anything else I can help you with?"
    assert result.budget.tool_calls_used == 0


def test_agent_tool_call_sets_proposed_action_without_running_executor() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    app = build_graph(_stub_tool_call(tool_call))
    state = AgentState(conversation=[Message(role="user", content="Where's my order?")])

    result = run(app, state)

    assert result.proposed_action == ProposedAction(
        id="call_1", tool_name="get_order_details", arguments={"order_id": "1"}
    )
    # the assistant's tool-call proposal is recorded in history (required for
    # correct tool_use/tool_result correlation on the next LLM call), but the
    # executor never ran: no tool *result* message was appended, budget untouched
    assert [m.role for m in result.conversation] == ["user", "assistant"]
    assert result.conversation[-1].tool_calls == [tool_call]
    assert result.budget.tool_calls_used == 0


def test_executor_path_dispatches_tool_and_increments_budget() -> None:
    app = build_graph(_stub_text("Thanks, all set."))
    state = AgentState(
        conversation=[Message(role="user", content="What's the status of my order?")],
        proposed_action=ProposedAction(
            id="call_1", tool_name="get_order_details", arguments={"order_id": "1"}
        ),
    )

    result = run(app, state)

    assert [m.role for m in result.conversation] == ["user", "tool", "assistant"]
    assert result.conversation[1].tool_call_id == "call_1"
    assert result.proposed_action is None
    assert result.budget.tool_calls_used == 1


def test_executor_dispatch_is_idempotent_for_a_retried_identical_state() -> None:
    """PLAN.md commit 17: a retried dispatch for the same (session_id, turn,
    action_hash) returns the cached result rather than dispatching again."""
    from guarded_agent.guardrails.write_gate import DispatchCache

    cache = DispatchCache()
    app = build_graph(_stub_text("Thanks, all set."), session_id="session-1", dispatch_cache=cache)
    state = AgentState(
        conversation=[Message(role="user", content="cancel it")],
        proposed_action=ProposedAction(
            id="call_1", tool_name="get_order_details", arguments={"order_id": "1"}
        ),
    )

    run(app, state)
    run(app, state)  # identical starting state -- same turn (budget.steps_used), same action

    assert len(cache._cache) == 1


def test_executor_path_records_unknown_tool_error_without_raising() -> None:
    app = build_graph(_stub_text("Sorry, something went wrong."))
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        proposed_action=ProposedAction(id="call_1", tool_name="delete_everything", arguments={}),
    )

    result = run(app, state)

    tool_message = result.conversation[-2]
    assert tool_message.role == "tool"
    assert "No tool registered" in (tool_message.content or "")


KILL_SWITCH_LIMITS = BudgetConfig(
    max_steps=5, max_tool_calls=5, max_tokens=1000, max_wall_clock_seconds=60.0
)


def _never_call(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
    raise AssertionError("generate_fn should never be called once budget is breached")


@pytest.mark.parametrize(
    "breached_budget",
    [
        BudgetCounters(steps_used=5),
        BudgetCounters(tool_calls_used=5),
        BudgetCounters(tokens_used=1000),
        BudgetCounters(elapsed_seconds=60.0),
    ],
)
def test_kill_switch_triggers_on_each_cap_without_calling_generate_fn(
    breached_budget: BudgetCounters,
) -> None:
    app = build_graph(_never_call, budget_limits=KILL_SWITCH_LIMITS)
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        budget=breached_budget,
    )

    result = run(app, state)

    assert result.escalated is True
    assert result.escalation_reason is not None
    assert result.conversation[-1].role == "assistant"


def test_kill_switch_takes_priority_over_a_pending_proposed_action() -> None:
    app = build_graph(_never_call, budget_limits=KILL_SWITCH_LIMITS)
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        proposed_action=ProposedAction(tool_name="get_order_details", arguments={"order_id": "1"}),
        budget=BudgetCounters(steps_used=5),
    )

    result = run(app, state)

    assert result.escalated is True


def test_repeated_tool_failure_triggers_escalation_without_calling_generate_fn() -> None:
    app = build_graph(_never_call, max_consecutive_tool_failures=2)
    state = AgentState(
        conversation=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="1", name="get_order_details", arguments={})],
            ),
            Message(role="tool", content="boom", tool_call_id="1", error=True),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="2", name="get_order_details", arguments={})],
            ),
            Message(role="tool", content="boom", tool_call_id="2", error=True),
        ]
    )

    result = run(app, state)

    assert result.escalated is True
    assert result.escalation_reason is not None
    assert "2 consecutive tool failures" in result.escalation_reason


def test_tool_failures_below_threshold_do_not_escalate() -> None:
    app = build_graph(_stub_text("Let me try something else."), max_consecutive_tool_failures=3)
    state = AgentState(
        conversation=[
            Message(role="user", content="hi"),
            Message(
                role="assistant",
                tool_calls=[ToolCall(id="1", name="get_order_details", arguments={})],
            ),
            Message(role="tool", content="boom", tool_call_id="1", error=True),
        ]
    )

    result = run(app, state)

    assert result.escalated is False


def test_policy_deadlock_triggers_escalation_without_calling_generate_fn() -> None:
    app = build_graph(_never_call, max_consecutive_policy_denials=2)
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        consecutive_policy_denials=2,
    )

    result = run(app, state)

    assert result.escalated is True
    assert result.escalation_reason is not None
    assert "2 consecutive policy denials" in result.escalation_reason


def test_policy_denials_below_threshold_do_not_escalate() -> None:
    app = build_graph(_stub_text("How can I help?"), max_consecutive_policy_denials=3)
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        consecutive_policy_denials=2,
    )

    result = run(app, state)

    assert result.escalated is False


_TRANSFER_TOOL_SCHEMA = {
    "type": "object",
    "properties": {"summary": {"type": "string"}},
    "required": ["summary"],
}


def _registry_with_transfer_tool() -> ToolRegistry:
    return ToolRegistry(
        {
            "transfer_to_human_agents": ToolDefinition(
                name="transfer_to_human_agents",
                description="Hand off the conversation to a human agent.",
                mutating=True,
                risk_tier="high",
                parameters=_TRANSFER_TOOL_SCHEMA,
            )
        }
    )


def test_escalation_proposes_transfer_tool_when_registry_has_it() -> None:
    app = build_graph(
        _never_call, registry=_registry_with_transfer_tool(), max_consecutive_policy_denials=1
    )
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        consecutive_policy_denials=1,
    )

    result = run(app, state)

    assert result.escalated is True
    assert result.proposed_action is not None
    assert result.proposed_action.tool_name == "transfer_to_human_agents"
    last_message = result.conversation[-1]
    assert last_message.role == "assistant"
    assert last_message.tool_calls is not None
    assert last_message.tool_calls[0].name == "transfer_to_human_agents"


def test_escalation_falls_back_to_text_when_transfer_tool_not_registered() -> None:
    # default registry (loaded from registry.yaml) has no transfer_to_human_agents tool
    app = build_graph(_never_call, max_consecutive_policy_denials=1)
    state = AgentState(
        conversation=[Message(role="user", content="hi")],
        consecutive_policy_denials=1,
    )

    result = run(app, state)

    assert result.escalated is True
    assert result.proposed_action is None
    last_message = result.conversation[-1]
    assert last_message.role == "assistant"
    assert last_message.tool_calls is None
    assert last_message.content is not None


class _StubRetriever:
    """Duck-typed stand-in for PolicyRetriever -- avoids loading the real
    embedding model in graph-wiring tests, which already have dedicated
    coverage in test_policy_retrieval.py."""

    def __init__(self, clauses: list[RetrievedClause]) -> None:
        self._clauses = clauses

    def retrieve(self, query: str, k: int) -> list[RetrievedClause]:
        return self._clauses[:k]


_A_CLAUSE = RetrievedClause(clause_id="c1", text="Some clause text.", score=0.9)


def _stub_policy_check(
    verdict: str, clause_id: str = "c1", reason: str = "because policy"
) -> PolicyCheckFn:
    def _check(action: ProposedAction, context: PolicyContext) -> PolicyVerdict:
        return PolicyVerdict(verdict=verdict, clause_id=clause_id, reason=reason)  # type: ignore[arg-type]

    return _check


def _never_check_policy(action: ProposedAction, context: PolicyContext) -> PolicyVerdict:
    raise AssertionError("policy_check_fn should never be called")


def _build_gated_graph(generate_fn: GenerateFn, policy_check_fn: PolicyCheckFn):
    return build_graph(
        generate_fn,
        policy_check_fn=policy_check_fn,
        retriever=_StubRetriever([_A_CLAUSE]),
        full_policy_text="FULL POLICY",
    )


def test_agent_text_reply_skips_policy_gate_entirely() -> None:
    app = _build_gated_graph(_stub_text("Here's your info."), _never_check_policy)
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.conversation[-1].content == "Here's your info."
    assert result.policy_verdict is None


def test_policy_gate_denies_invalid_schema_without_calling_policy_check_fn() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={})  # missing order_id
    app = _build_gated_graph(_stub_tool_call(tool_call), _never_check_policy)
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.proposed_action is None
    assert result.conversation[-1].role == "assistant"
    assert result.policy_verdict is None  # never reached the policy check


def test_policy_gate_denies_based_on_policy_verdict() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    app = _build_gated_graph(
        _stub_tool_call(tool_call), _stub_policy_check("DENY", reason="not allowed")
    )
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.proposed_action is None
    assert result.policy_verdict is not None
    assert result.policy_verdict.verdict == "DENY"
    assert "not allowed" in (result.conversation[-1].content or "")


def test_allowed_non_mutating_action_needs_no_confirmation() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    app = _build_gated_graph(_stub_tool_call(tool_call), _stub_policy_check("ALLOW"))
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.proposed_action is not None  # ready to send, untouched by the write gate
    assert result.pending_confirmation is None
    assert result.policy_verdict is not None
    assert result.policy_verdict.verdict == "ALLOW"


def test_allowed_mutating_action_requires_confirmation() -> None:
    tool_call = ToolCall(
        id="call_1", name="issue_refund", arguments={"order_id": "1", "amount": 10.0}
    )
    app = _build_gated_graph(_stub_tool_call(tool_call), _stub_policy_check("ALLOW"))
    state = AgentState(conversation=[Message(role="user", content="refund me")])

    result = run(app, state)

    assert result.proposed_action is None  # not sent out yet
    assert result.pending_confirmation is not None
    assert result.pending_confirmation.tool_name == "issue_refund"
    assert "issue_refund" in (result.conversation[-1].content or "")


def test_confirmation_lets_the_same_mutating_action_through() -> None:
    tool_call = ToolCall(
        id="call_2", name="issue_refund", arguments={"order_id": "1", "amount": 10.0}
    )
    app = _build_gated_graph(_stub_tool_call(tool_call), _stub_policy_check("ALLOW"))
    state = AgentState(
        conversation=[
            Message(role="user", content="refund me"),
            Message(role="assistant", content="Shall I proceed? (yes/no)"),
            Message(role="user", content="yes"),
        ],
        pending_confirmation=ProposedAction(
            id="call_1", tool_name="issue_refund", arguments={"order_id": "1", "amount": 10.0}
        ),
    )

    result = run(app, state)

    assert result.proposed_action is not None  # now ready to send
    assert result.pending_confirmation is None


def test_needs_confirmation_verdict_gates_even_a_non_mutating_tool() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    app = _build_gated_graph(_stub_tool_call(tool_call), _stub_policy_check("NEEDS_CONFIRMATION"))
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.proposed_action is None
    assert result.pending_confirmation is not None


def _stub_critic_check(approved: bool, reason: str = "because critic") -> CriticCheckFn:
    def _check(conversation: list[Message], draft: str, context: PolicyContext) -> CriticVerdict:
        return CriticVerdict(approved=approved, reason=reason)

    return _check


def _stub_critic_sequence(verdicts: list[tuple[bool, str]]) -> CriticCheckFn:
    """Same idea as _stub_text_sequence, for the critic side of a
    reject-then-approve revision test."""
    calls = {"n": 0}

    def _check(conversation: list[Message], draft: str, context: PolicyContext) -> CriticVerdict:
        approved, reason = verdicts[min(calls["n"], len(verdicts) - 1)]
        calls["n"] += 1
        return CriticVerdict(approved=approved, reason=reason)

    return _check


def _never_check_critic(
    conversation: list[Message], draft: str, context: PolicyContext
) -> CriticVerdict:
    raise AssertionError("critic_check_fn should never be called")


def _build_gated_graph_with_critic(generate_fn: GenerateFn, critic_check_fn: CriticCheckFn):
    return build_graph(
        generate_fn,
        policy_check_fn=_never_check_policy,
        retriever=_StubRetriever([_A_CLAUSE]),
        full_policy_text="FULL POLICY",
        critic_check_fn=critic_check_fn,
    )


def test_critic_approves_draft_and_it_goes_out_unchanged() -> None:
    app = _build_gated_graph_with_critic(
        _stub_text("Here's your order status."), _stub_critic_check(approved=True)
    )
    state = AgentState(conversation=[Message(role="user", content="Where's my order?")])

    result = run(app, state)

    assert result.conversation[-1].content == "Here's your order status."
    assert result.escalated is False
    assert result.critic_feedback is None


def test_critic_rejects_draft_triggers_one_revision_then_proceeds() -> None:
    generate_fn = _stub_text_sequence(
        ["Your refund was already sent.", "Let me look into that for you."]
    )
    critic_check_fn = _stub_critic_sequence([(False, "unsupported claim"), (True, "grounded now")])
    app = _build_gated_graph_with_critic(generate_fn, critic_check_fn)
    state = AgentState(conversation=[Message(role="user", content="Where's my refund?")])

    result = run(app, state)

    assert result.escalated is False
    assert result.conversation[-1].content == "Let me look into that for you."
    assert result.critic_feedback is None
    # the rejected first draft never made it into history -- the user never saw it
    assistant_contents = [m.content for m in result.conversation if m.role == "assistant"]
    assert "Your refund was already sent." not in assistant_contents


def test_critic_bounded_retry_escalates_after_exactly_one_revision() -> None:
    """PLAN.md commit 20's literal acceptance check: the critic loop must
    not exceed one revision, no matter how many times the critic rejects."""
    call_count = {"n": 0}

    def _always_draft(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        call_count["n"] += 1
        return AgentDecision(content=f"draft #{call_count['n']}")

    def _always_reject(
        conversation: list[Message], draft: str, context: PolicyContext
    ) -> CriticVerdict:
        return CriticVerdict(approved=False, reason="never good enough")

    app = _build_gated_graph_with_critic(_always_draft, _always_reject)
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert call_count["n"] == 2  # exactly one initial draft plus one bounded revision, never more
    assert result.escalated is True
    assert result.escalation_reason is not None
    assert "critic" in result.escalation_reason
    assert result.critic_feedback is None  # cleared even on the escalation path


def test_revision_with_no_usable_text_escalates_without_a_second_critic_pass() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})

    def _text_then_tool_call(
        conversation: list[Message], tools: list[ToolDefinition]
    ) -> AgentDecision:
        if any("Internal note" in (m.content or "") for m in conversation):
            return AgentDecision(tool_call=tool_call)
        return AgentDecision(content="Your refund was already sent.")

    app = _build_gated_graph_with_critic(_text_then_tool_call, _stub_critic_check(approved=False))
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.escalated is True
    assert "not usable" in (result.escalation_reason or "")


def test_tool_call_proposal_skips_critic_entirely() -> None:
    tool_call = ToolCall(id="call_1", name="get_order_details", arguments={"order_id": "1"})
    app = build_graph(
        _stub_tool_call(tool_call),
        policy_check_fn=_stub_policy_check("ALLOW"),
        retriever=_StubRetriever([_A_CLAUSE]),
        full_policy_text="FULL POLICY",
        critic_check_fn=_never_check_critic,
    )
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.proposed_action is not None


def test_empty_decision_escalation_skips_critic() -> None:
    def _empty_decision(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        return AgentDecision()

    app = _build_gated_graph_with_critic(_empty_decision, _never_check_critic)
    state = AgentState(conversation=[Message(role="user", content="hi")])

    result = run(app, state)

    assert result.escalated is True
    assert result.escalation_reason == "agent produced neither a reply nor a tool call"


@pytest.mark.skipif(
    not MOCK_TASKS_PATH.exists(),
    reason="vendored tau2-bench data not present -- run `make install` first",
)
def test_mock_domain_smoke_run_completes() -> None:
    """Seeded from a real tau2 mock-domain task, not a synthetic string.

    Uses a stub generate_fn -- no live LLM call in the test suite, per
    CLAUDE.md's fixture-based testing convention. The real integration is
    verified live via `make smoke-guarded` (PLAN.md commit 11), not here.
    """
    tasks = json.loads(MOCK_TASKS_PATH.read_text(encoding="utf-8"))
    ticket = tasks[0]["ticket"]

    app = build_graph(_stub_text("Done -- I've created that task for you."))
    state = AgentState(conversation=[Message(role="user", content=ticket)])

    result = run(app, state)

    assert result.conversation[-1].role == "assistant"
    assert result.escalated is False
