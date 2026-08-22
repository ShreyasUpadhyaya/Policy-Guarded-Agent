from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from functools import lru_cache
from typing import Any

from tau2.agent.base.llm_config import LLMConfigMixin
from tau2.agent.base_agent import HalfDuplexAgent, ValidAgentInputMessage
from tau2.data_model.message import AssistantMessage, MultiToolMessage
from tau2.data_model.message import Message as Tau2Message
from tau2.data_model.message import SystemMessage as Tau2SystemMessage
from tau2.data_model.message import ToolCall as Tau2ToolCall
from tau2.data_model.message import ToolMessage as Tau2ToolMessage
from tau2.data_model.message import UserMessage as Tau2UserMessage
from tau2.environment.tool import Tool as Tau2Tool
from tau2.registry import registry as tau2_registry
from tau2.utils.llm_utils import generate as tau2_generate

from guarded_agent.config import load_config
from guarded_agent.graph import AgentDecision, GenerateFn, build_graph
from guarded_agent.graph import run as run_graph
from guarded_agent.guardrails.critic import make_llm_critic_check_fn
from guarded_agent.guardrails.policy_checker import make_llm_policy_check_fn
from guarded_agent.guardrails.policy_retrieval import PolicyContext, PolicyRetriever
from guarded_agent.state import AgentState, Message, PolicyVerdict, ProposedAction, ToolCall
from guarded_agent.telemetry.tracing import configure_tracing
from guarded_agent.tools.registry import ToolDefinition, ToolRegistry

AGENT_NAME = "guarded_agent"

GUARDRAIL_STAGES = ("registry", "policy_checker", "critic")
"""Cumulative ablation stages (PLAN.md commit 23's five variants: baseline,
+registry, +policy_checker, +critic, full). Budget enforcement and
escalation are deliberately not one of these -- they're safety
infrastructure that predates Day 3's guardrails (commit 12's kill switch),
not a guardrail feature the ablation study is measuring, so every variant
including baseline keeps them."""

AGENT_INSTRUCTION = """
You are a customer service agent that helps the user according to the <policy> provided below.
In each turn you can either send a message to the user or make a tool call, not both.
Always follow the policy exactly. Always produce valid JSON for tool calls.
""".strip()

SYSTEM_PROMPT_TEMPLATE = """
<instructions>
{instruction}
</instructions>
<policy>
{domain_policy}
</policy>
""".strip()


def _to_our_messages(message: Tau2Message) -> list[Message]:
    """Convert one tau2 message into one or more of our Message objects.

    A MultiToolMessage (results from several tool calls proposed in one
    turn) expands into several tool-role Messages.
    """
    if isinstance(message, Tau2UserMessage):
        return [Message(role="user", content=message.content)]
    if isinstance(message, AssistantMessage):
        tool_calls = (
            [ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments) for tc in message.tool_calls]
            if message.tool_calls
            else None
        )
        return [Message(role="assistant", content=message.content, tool_calls=tool_calls)]
    if isinstance(message, MultiToolMessage):
        return [
            Message(role="tool", content=tm.content, tool_call_id=tm.id, error=tm.error)
            for tm in message.tool_messages
        ]
    if isinstance(message, Tau2ToolMessage):
        return [
            Message(
                role="tool", content=message.content, tool_call_id=message.id, error=message.error
            )
        ]
    raise TypeError(f"Unsupported tau2 message type: {type(message).__name__}")


def _to_tau2_messages(conversation: list[Message], system_prompt: str) -> list[Tau2Message]:
    tau2_messages: list[Tau2Message] = [Tau2SystemMessage(role="system", content=system_prompt)]
    for message in conversation:
        if message.role == "user":
            tau2_messages.append(Tau2UserMessage(role="user", content=message.content))
        elif message.role == "assistant":
            tool_calls = (
                [
                    Tau2ToolCall(id=tc.id, name=tc.name, arguments=tc.arguments)
                    for tc in message.tool_calls
                ]
                if message.tool_calls
                else None
            )
            tau2_messages.append(
                AssistantMessage(role="assistant", content=message.content, tool_calls=tool_calls)
            )
        else:  # "tool"
            tau2_messages.append(
                Tau2ToolMessage(id=message.tool_call_id or "", role="tool", content=message.content)
            )
    return tau2_messages


@lru_cache(maxsize=8)
def _cached_retriever(domain_policy: str) -> PolicyRetriever:
    """Building a PolicyRetriever loads the embedding model and indexes the
    policy text (~1-2s). tau2 constructs a fresh GuardedTau2Agent per task,
    but every task in a domain shares the same domain_policy string, so
    caching by that text avoids rebuilding the same index for every task in
    a run. maxsize=8 is enough headroom for every tau2 domain at once
    without the cache growing unbounded across a long process lifetime.
    """
    return PolicyRetriever.from_policy_text(domain_policy)


def _extract_mutating_by_name(tools: list[Tau2Tool]) -> dict[str, bool]:
    """Read each tool's real read/write classification off its underlying
    function, set by tau2's own `@is_tool(ToolType.WRITE)` decorator.

    Not part of tau2's advertised public API -- `Tool._func` is a private
    attribute -- but stable for our purposes since tau2 is pinned to an
    exact tag (v1.0.1). Falls back to treating a tool as mutating if the
    attribute is somehow missing: fail-safe forcing, same as
    ToolRegistry.from_openai_schemas's own default.
    """
    return {tool.name: getattr(tool._func, "__mutates_state__", True) for tool in tools}


class TurnCost:
    """Mutable, per-turn cost/usage accumulator shared by every real LLM
    call made while producing one outgoing message: the main agent
    generation, plus any policy_checker/critic calls the graph triggers for
    the same turn (including a critic-rejected first draft, which calls
    generate_fn a second time via agent_revise). A single turn can involve
    several real, separately-billed LLM calls; tau2's own cost aggregation
    (get_cost()) sums each *message's* cost across the conversation, so this
    must hold the sum of everything spent this turn, not just the last call
    -- and must be reset at the start of every turn (generate_next_message)
    or costs would compound across turns instead of resetting to what this
    turn alone spent.

    tau2's own trace/cost aggregation reads .cost/.usage off the outgoing
    AssistantMessage, which our simplified AgentDecision (deliberately
    tau2-independent) doesn't carry all the way through -- this is the side
    channel that closes that gap.
    """

    def __init__(self) -> None:
        self.cost: float = 0.0
        self.usage: dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0}

    def add(self, cost: float | None, usage: dict[str, int] | None) -> None:
        self.cost += cost or 0.0
        if usage:
            self.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
            self.usage["completion_tokens"] += usage.get("completion_tokens", 0)

    def reset(self) -> None:
        self.cost = 0.0
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}


def make_tau2_generate_fn(
    tau2_tools: list[Tau2Tool],
    model: str,
    llm_args: dict[str, Any],
    domain_policy: str,
    turn_cost: TurnCost,
) -> GenerateFn:
    """Real LLM generation, reusing tau2's own generate() helper.

    Deliberately reuses tau2's well-tested message/tool-schema formatting
    and retry logic rather than reimplementing OpenAI-style tool-calling
    from scratch (PLAN.md commit 11 notes). If the model proposes more than
    one tool call in a single turn, only the first is used -- a known,
    accepted limitation rather than a full multi-call redesign.
    """
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        instruction=AGENT_INSTRUCTION, domain_policy=domain_policy
    )

    def _generate(conversation: list[Message], tools: list[ToolDefinition]) -> AgentDecision:
        tau2_messages = _to_tau2_messages(conversation, system_prompt)
        response = tau2_generate(
            model=model,
            messages=tau2_messages,
            tools=tau2_tools,
            call_name="guarded_agent_response",
            **llm_args,
        )
        turn_cost.add(response.cost, response.usage)
        usage = response.usage or {}
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        if response.is_tool_call():
            assert response.tool_calls is not None
            call = response.tool_calls[0]
            return AgentDecision(
                tool_call=ToolCall(id=call.id, name=call.name, arguments=call.arguments),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        return AgentDecision(
            content=response.content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    return _generate


class GuardedTau2Agent(LLMConfigMixin, HalfDuplexAgent[AgentState]):  # type: ignore[misc]
    """Bridges tau2's HalfDuplexAgent protocol to our own LangGraph graph.

    Translates tau2 Message types <-> guarded_agent.state.Message at the
    boundary; the graph itself (guarded_agent.graph) never imports tau2.
    """

    def __init__(
        self,
        tools: list[Tau2Tool],
        domain_policy: str,
        llm: str,
        llm_args: dict[str, Any] | None = None,
        enabled_guardrails: frozenset[str] = frozenset(GUARDRAIL_STAGES),
    ) -> None:
        configure_tracing()
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        registry = ToolRegistry.from_openai_schemas(
            [t.openai_schema for t in tools],
            mutating_by_name=_extract_mutating_by_name(tools),
        )
        self._turn_cost = TurnCost()
        generate_fn = make_tau2_generate_fn(
            tools, llm, self.llm_args, domain_policy, self._turn_cost
        )
        run_config = load_config()

        graph_kwargs: dict[str, Any] = {}
        if "registry" in enabled_guardrails:
            graph_kwargs["policy_check_fn"] = (
                make_llm_policy_check_fn(llm, on_cost=self._turn_cost.add)
                if "policy_checker" in enabled_guardrails
                else _always_allow_policy_check
            )
            graph_kwargs["retriever"] = _cached_retriever(domain_policy)
            graph_kwargs["full_policy_text"] = domain_policy
            graph_kwargs["retrieval_top_k"] = run_config.retrieval.top_k
            graph_kwargs["retrieval_min_confidence"] = run_config.retrieval.min_confidence
            if "critic" in enabled_guardrails:
                graph_kwargs["critic_check_fn"] = make_llm_critic_check_fn(
                    llm, on_cost=self._turn_cost.add
                )

        self.app = build_graph(
            generate_fn,
            registry,
            run_config.budget,
            session_id=str(uuid.uuid4()),
            max_consecutive_tool_failures=run_config.escalation.max_consecutive_tool_failures,
            max_consecutive_policy_denials=run_config.escalation.max_consecutive_policy_denials,
            **graph_kwargs,
        )
        self._started_at = time.time()

    def get_init_state(self, message_history: list[Tau2Message] | None = None) -> AgentState:
        self._started_at = time.time()
        state = AgentState()
        for message in message_history or []:
            for our_message in _to_our_messages(message):
                state = state.add_message(our_message)
        return state

    def generate_next_message(
        self, message: ValidAgentInputMessage, state: AgentState
    ) -> tuple[AssistantMessage, AgentState]:
        for our_message in _to_our_messages(message):
            state = state.add_message(our_message)

        elapsed = time.time() - self._started_at
        state = state.model_copy(update={"budget": state.budget.with_elapsed(elapsed)})

        # Reset before running the graph, not after: this turn's accumulator
        # must start empty regardless of how the *previous* turn ended, and
        # must capture every real LLM call the graph makes below (agent,
        # policy_checker, critic, and agent_revise's second agent call on a
        # critic rejection all land in the same TurnCost instance). Without
        # this reset, an escalation reached without calling the LLM this turn
        # (e.g. a budget breach caught by entry_router before "agent" even
        # runs) would misattribute a *previous* turn's cost to a free message
        # -- the bug an earlier version of this method avoided by forcing
        # escalated turns to always report 0.0, which had its own cost: it
        # also zeroed out turns that escalated *after* real spend this turn
        # (e.g. a second critic rejection, which calls agent + critic +
        # agent_revise + critic_final before giving up). Resetting per turn
        # and always reporting the accumulator's real total fixes both:
        # verified live (PLAN.md commit 21 v2 smoke run) that a None cost
        # poisons tau2's get_cost() for the whole conversation, and
        # TurnCost.cost is a plain float that starts at 0.0, so it can never
        # be None.
        self._turn_cost.reset()
        state = run_graph(self.app, state)
        cost = self._turn_cost.cost
        usage = self._turn_cost.usage

        if state.proposed_action is not None:
            action = state.proposed_action
            assistant_message = AssistantMessage(
                role="assistant",
                tool_calls=[
                    Tau2ToolCall(id=action.id, name=action.tool_name, arguments=action.arguments)
                ],
                cost=cost,
                usage=usage,
            )
            # The proposal is now consumed (translated into the outgoing message);
            # tau2 executes it externally and will hand the result back as the
            # *next* turn's input. If we left this set, the next call's router
            # would send us to our own executor and double-record a tool result.
            state = state.model_copy(update={"proposed_action": None})
        else:
            assistant_message = AssistantMessage(
                role="assistant", content=state.conversation[-1].content, cost=cost, usage=usage
            )

        return assistant_message, state


def _always_allow_policy_check(
    conversation: list[Message], action: ProposedAction, context: PolicyContext
) -> PolicyVerdict:
    """Ablation-only stand-in for a real policy checker (PLAN.md commit 23's
    '+registry' variant): schema validation and write_gate's real
    mutating-based confirmation still run for real, but nothing here makes
    an actual LLM-based policy compliance judgment yet. Never used by the
    default/production agent -- only by the '+registry' ablation factory,
    which explicitly asks for it via enabled_guardrails.
    """
    return PolicyVerdict(
        verdict="ALLOW",
        clause_id="ablation-stub",
        reason="policy checker disabled for this ablation variant",
    )


def create_guarded_agent(
    tools: list[Tau2Tool], domain_policy: str, **kwargs: Any
) -> GuardedTau2Agent:
    """Factory function, matching tau2's `factory(tools, domain_policy, **kwargs)` pattern.

    Always builds the "full" variant (every guardrail enabled) -- this is
    the production/default agent, registered under AGENT_NAME, and its
    behavior must stay exactly what every prior commit already verified.
    See _make_ablation_agent_factory for the other four ablation variants.
    """
    llm = kwargs.get("llm")
    if llm is None:
        raise ValueError("create_guarded_agent requires 'llm' (pass --agent-llm on the CLI)")
    return GuardedTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=kwargs.get("llm_args"),
    )


def _make_ablation_agent_factory(
    enabled_guardrails: frozenset[str],
) -> Callable[..., GuardedTau2Agent]:
    """Build a tau2 agent factory pinned to a fixed set of enabled
    guardrails (PLAN.md commit 23).

    tau2's own agent-construction call (tau2/runner/build.py) only ever
    passes a fixed set of kwargs (llm, llm_args, task, ...) to whichever
    factory is registered under the name evals/ablations.yaml's `agent`
    field selects -- there's no channel to pass "which guardrails" through
    that call. So each ablation variant gets its own registered factory
    name instead, with enabled_guardrails baked in via this closure, rather
    than smuggled through llm_args.
    """

    def factory(tools: list[Tau2Tool], domain_policy: str, **kwargs: Any) -> GuardedTau2Agent:
        llm = kwargs.get("llm")
        if llm is None:
            raise ValueError("this agent factory requires 'llm' (pass --agent-llm on the CLI)")
        return GuardedTau2Agent(
            tools=tools,
            domain_policy=domain_policy,
            llm=llm,
            llm_args=kwargs.get("llm_args"),
            enabled_guardrails=enabled_guardrails,
        )

    return factory


ABLATION_AGENT_VARIANTS: dict[str, frozenset[str]] = {
    "guarded_agent_baseline": frozenset(),
    "guarded_agent_registry": frozenset({"registry"}),
    "guarded_agent_policy_checker": frozenset({"registry", "policy_checker"}),
    "guarded_agent_critic": frozenset({"registry", "policy_checker", "critic"}),
}
"""The four ablation stages short of "full" (PLAN.md commit 23). "full"
itself is AGENT_NAME ("guarded_agent") -- the existing, unchanged default
factory -- since there's no fifth guardrail left to add on top of critic."""


def register() -> None:
    """Register our agent factories with tau2's registry.

    Safe to call more than once (e.g. if this module is imported from
    multiple entry points) -- tau2_registry.register_agent_factory raises on
    a duplicate name, so each registration is guarded individually.
    """
    if tau2_registry.get_agent_factory(AGENT_NAME) is None:
        tau2_registry.register_agent_factory(create_guarded_agent, AGENT_NAME)
    for name, enabled in ABLATION_AGENT_VARIANTS.items():
        if tau2_registry.get_agent_factory(name) is None:
            tau2_registry.register_agent_factory(_make_ablation_agent_factory(enabled), name)


register()
