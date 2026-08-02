from __future__ import annotations

import time
import uuid
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
from guarded_agent.guardrails.policy_checker import make_llm_policy_check_fn
from guarded_agent.guardrails.policy_retrieval import PolicyRetriever
from guarded_agent.state import AgentState, Message, ToolCall
from guarded_agent.telemetry.tracing import configure_tracing
from guarded_agent.tools.registry import ToolDefinition, ToolRegistry

AGENT_NAME = "guarded_agent"

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
            Message(role="tool", content=tm.content, tool_call_id=tm.id)
            for tm in message.tool_messages
        ]
    if isinstance(message, Tau2ToolMessage):
        return [Message(role="tool", content=message.content, tool_call_id=message.id)]
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


class LastGeneration:
    """Mutable side-channel holding the most recent tau2 generate() call's
    cost/usage, so generate_next_message can attach real figures to the
    outgoing AssistantMessage -- tau2's own trace and cost aggregation reads
    .cost/.usage off the message itself, which our simplified AgentDecision
    (deliberately tau2-independent) doesn't carry all the way through.
    """

    def __init__(self) -> None:
        self.cost: float | None = None
        self.usage: dict[str, int] | None = None


def make_tau2_generate_fn(
    tau2_tools: list[Tau2Tool],
    model: str,
    llm_args: dict[str, Any],
    domain_policy: str,
    last_generation: LastGeneration,
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
        last_generation.cost = response.cost
        last_generation.usage = response.usage
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
    ) -> None:
        configure_tracing()
        super().__init__(tools=tools, domain_policy=domain_policy, llm=llm, llm_args=llm_args)
        registry = ToolRegistry.from_openai_schemas(
            [t.openai_schema for t in tools],
            mutating_by_name=_extract_mutating_by_name(tools),
        )
        self._last_generation = LastGeneration()
        generate_fn = make_tau2_generate_fn(
            tools, llm, self.llm_args, domain_policy, self._last_generation
        )
        run_config = load_config()
        self.app = build_graph(
            generate_fn,
            registry,
            run_config.budget,
            policy_check_fn=make_llm_policy_check_fn(llm),
            retriever=_cached_retriever(domain_policy),
            full_policy_text=domain_policy,
            retrieval_top_k=run_config.retrieval.top_k,
            retrieval_min_confidence=run_config.retrieval.min_confidence,
            session_id=str(uuid.uuid4()),
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

        state = run_graph(self.app, state)

        # The kill switch (budget breach) produces a message without calling the
        # LLM this turn -- attaching self._last_generation's cost/usage there
        # would misattribute a *previous* turn's figures to a free message.
        if state.escalated:
            cost, usage = None, None
        else:
            cost = self._last_generation.cost
            usage = self._last_generation.usage

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


def create_guarded_agent(
    tools: list[Tau2Tool], domain_policy: str, **kwargs: Any
) -> GuardedTau2Agent:
    """Factory function, matching tau2's `factory(tools, domain_policy, **kwargs)` pattern."""
    llm = kwargs.get("llm")
    if llm is None:
        raise ValueError("create_guarded_agent requires 'llm' (pass --agent-llm on the CLI)")
    return GuardedTau2Agent(
        tools=tools,
        domain_policy=domain_policy,
        llm=llm,
        llm_args=kwargs.get("llm_args"),
    )


def register() -> None:
    """Register our agent factory with tau2's registry.

    Safe to call more than once (e.g. if this module is imported from
    multiple entry points) -- tau2_registry.register_agent_factory raises on
    a duplicate name, so this guards against that.
    """
    if tau2_registry.get_agent_factory(AGENT_NAME) is None:
        tau2_registry.register_agent_factory(create_guarded_agent, AGENT_NAME)


register()
