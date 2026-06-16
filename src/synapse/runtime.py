"""The agent loop: drive a model + tools to completion (async-first).

This is the single place that orchestrates "call model → run tools → feed
results back → repeat". It also threads the v0.2 capabilities through one
:class:`RunContext`: observability hooks, tool approval (HITL), input/output
guardrails, a verifier (iterate-until-pass), token budgets, context compaction,
checkpointing, and tool search.

Concurrency: a turn's tool calls (including sub-agent delegations) run
concurrently via :func:`asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Coroutine, Optional, TypeVar, Union

from ._util import maybe_await
from .guardrails import Guardrail, apply_guardrails
from .messages import Message, ToolResultBlock, ToolUseBlock
from .observability import Hooks, Usage
from .tool import Tool

if TYPE_CHECKING:  # avoid circular imports at runtime
    from .agent import Agent
    from .checkpoint import Checkpointer
    from .context import Compactor

_T = TypeVar("_T")


@dataclass
class ApprovalDecision:
    """Result of a human-in-the-loop approval check."""

    allow: bool
    reason: str = ""


@dataclass
class Verdict:
    """Result of a verifier check on an agent's output."""

    passed: bool
    feedback: str = ""


# Callbacks may return their value directly or as a coroutine.
ApprovalCallback = Callable[[str, dict], Union[bool, ApprovalDecision, Awaitable[Any]]]
Verifier = Callable[[str], Union[bool, Verdict, Awaitable[Any]]]


@dataclass
class Session:
    """Carries conversation history across multiple turns with an agent.

    A per-session lock serializes turns on the *same* session; distinct
    sessions run fully in parallel.
    """

    messages: list[Message] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def extend(self, messages: list[Message]) -> None:
        self.messages = list(messages)


@dataclass
class RunResult:
    """The outcome of running an agent once."""

    output: str
    messages: list[Message]
    agent: str
    iterations: int
    stop_reason: str = "end_turn"
    usage: Usage = field(default_factory=Usage)
    verify_rounds: int = 1


@dataclass
class RunContext:
    """Everything the loop needs beyond the agent itself."""

    max_iterations: int = 12
    hooks: Optional[Hooks] = None
    approval: Optional[ApprovalCallback] = None
    verify: Optional[Verifier] = None
    max_verify_rounds: int = 3
    token_budget: Optional[int] = None
    input_guardrails: list[Guardrail] = field(default_factory=list)
    output_guardrails: list[Guardrail] = field(default_factory=list)
    compactor: Optional["Compactor"] = None
    checkpointer: Optional["Checkpointer"] = None
    run_id: Optional[str] = None
    tool_search: bool = False
    usage: Usage = field(default_factory=Usage)


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from synchronous code, anywhere."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    box: dict[str, Any] = {}

    def _runner() -> None:
        box["value"] = asyncio.run(coro)

    thread = threading.Thread(target=_runner)
    thread.start()
    thread.join()
    return box["value"]


async def _emit(ctx: RunContext, method: str, *args: Any) -> None:
    if ctx.hooks is not None:
        await maybe_await(getattr(ctx.hooks, method)(*args))


def _normalize_approval(value: Any) -> ApprovalDecision:
    if isinstance(value, ApprovalDecision):
        return value
    return ApprovalDecision(allow=bool(value))


def _normalize_verdict(value: Any) -> Verdict:
    if isinstance(value, Verdict):
        return value
    if isinstance(value, tuple):
        passed, feedback = value
        return Verdict(passed=bool(passed), feedback=feedback)
    return Verdict(passed=bool(value))


def _make_search_tool(agent: "Agent", active: set[str]) -> Tool:
    from .context import select_tools

    def search_tools(query: str) -> str:
        matches = select_tools(query, agent.tools, k=5)
        for t in matches:
            active.add(t.name)
        if not matches:
            return "no matching tools"
        return "Activated tools:\n" + "\n".join(f"- {t.name}: {t.description}" for t in matches)

    return Tool(
        name="search_tools",
        description="Search available tools by capability; matches become callable this run.",
        parameters={
            "type": "object",
            "properties": {"query": {"type": "string", "description": "Capability to find."}},
            "required": ["query"],
        },
        func=search_tools,
    )


async def _execute_tool_use(
    ctx: RunContext,
    tool_map: dict[str, Tool],
    block: ToolUseBlock,
) -> ToolResultBlock:
    impl = tool_map.get(block.name)
    if impl is None:
        return ToolResultBlock(block.id, f"Error: no tool named {block.name!r}", is_error=True)

    if impl.requires_approval and ctx.approval is not None:
        decision = _normalize_approval(
            await maybe_await(ctx.approval(block.name, block.input))
        )
        if not decision.allow:
            reason = decision.reason or "denied by approval policy"
            return ToolResultBlock(block.id, f"Tool call denied: {reason}", is_error=True)

    await _emit(ctx, "on_tool_start", block.name, block.input)
    try:
        result = await impl.invoke(**block.input)
        out = ToolResultBlock(block.id, "" if result is None else str(result))
    except Exception as exc:  # tools surface failures back to the model
        out = ToolResultBlock(block.id, f"Error: {exc}", is_error=True)
    await _emit(ctx, "on_tool_end", block.name, out.content, out.is_error)
    return out


async def _drive(
    ctx: RunContext,
    agent: "Agent",
    messages: list[Message],
) -> tuple[int, str]:
    active: set[str] = set()
    base_map = agent.tool_map
    search_tool = _make_search_tool(agent, active) if ctx.tool_search else None

    def exposed_tools() -> list[Tool]:
        if search_tool is None:
            return agent.tools
        return [search_tool, *[t for t in agent.tools if t.name in active]]

    def tool_map() -> dict[str, Tool]:
        if search_tool is None:
            return base_map
        return {search_tool.name: search_tool, **base_map}

    stop_reason = "end_turn"
    iterations = 0
    for iterations in range(1, ctx.max_iterations + 1):
        if ctx.compactor is not None:
            messages[:] = await ctx.compactor.maybe_compact(messages)

        response = await agent.model.generate(
            system=agent.instructions,
            messages=messages,
            tools=exposed_tools(),
        )
        messages.append(response.message)
        ctx.usage.add(response.usage)
        stop_reason = response.stop_reason
        await _emit(ctx, "on_model_response", response)
        await _emit(ctx, "on_turn_end", iterations, response.message)

        if ctx.checkpointer is not None and ctx.run_id is not None:
            await ctx.checkpointer.save(ctx.run_id, messages)

        if stop_reason != "tool_use":
            return iterations, stop_reason

        if ctx.token_budget is not None and ctx.usage.total_tokens >= ctx.token_budget:
            return iterations, "budget_exceeded"

        results = await asyncio.gather(
            *(_execute_tool_use(ctx, tool_map(), block) for block in response.message.tool_uses)
        )
        messages.append(Message(role="user", content=list(results)))

        if ctx.checkpointer is not None and ctx.run_id is not None:
            await ctx.checkpointer.save(ctx.run_id, messages)

    return iterations, "max_iterations"


async def _run_with_verify(
    ctx: RunContext,
    agent: "Agent",
    messages: list[Message],
) -> tuple[int, str, int]:
    total_iterations = 0
    stop_reason = "end_turn"
    rounds = 0
    for rounds in range(1, ctx.max_verify_rounds + 1):
        iters, stop_reason = await _drive(ctx, agent, messages)
        total_iterations += iters
        if ctx.verify is None:
            break
        output = next((m.text for m in reversed(messages) if m.role == "assistant"), "")
        verdict = _normalize_verdict(await maybe_await(ctx.verify(output)))
        if verdict.passed:
            stop_reason = "verified"
            break
        if rounds < ctx.max_verify_rounds:
            messages.append(
                Message(
                    role="user",
                    content=(
                        "Your previous answer was rejected by verification: "
                        f"{verdict.feedback or 'does not meet requirements'}. Please revise."
                    ),
                )
            )
        else:
            stop_reason = "verification_failed"
    return total_iterations, stop_reason, rounds


async def arun_agent(
    agent: "Agent",
    user_input: str,
    *,
    max_iterations: int = 12,
    session: Session | None = None,
    context: RunContext | None = None,
) -> RunResult:
    """Run ``agent`` on ``user_input`` until done (async).

    Pass a :class:`RunContext` to enable hooks, approval, verification,
    guardrails, budgets, compaction, checkpointing, or tool search.
    """
    ctx = context or RunContext(max_iterations=max_iterations)
    if context is None:
        ctx.max_iterations = max_iterations

    await _emit(ctx, "on_run_start", agent.name, user_input)
    cleaned = await apply_guardrails(user_input, ctx.input_guardrails)

    async def _body() -> tuple[list[Message], int, str, int]:
        messages: list[Message] = []
        if session is not None:
            messages = list(session.messages)
        elif ctx.checkpointer is not None and ctx.run_id is not None:
            restored = await ctx.checkpointer.load(ctx.run_id)
            if restored:
                messages = restored
        messages.append(Message(role="user", content=cleaned))
        iters, stop, rounds = await _run_with_verify(ctx, agent, messages)
        if session is not None:
            session.extend(messages)
        return messages, iters, stop, rounds

    if session is not None:
        async with session.lock:
            messages, iterations, stop_reason, rounds = await _body()
    else:
        messages, iterations, stop_reason, rounds = await _body()

    output = next((m.text for m in reversed(messages) if m.role == "assistant"), "")
    output = await apply_guardrails(output, ctx.output_guardrails)

    result = RunResult(
        output=output,
        messages=messages,
        agent=agent.name,
        iterations=iterations,
        stop_reason=stop_reason,
        usage=ctx.usage,
        verify_rounds=rounds,
    )
    await _emit(ctx, "on_run_end", result)
    return result


def run_agent(
    agent: "Agent",
    user_input: str,
    *,
    max_iterations: int = 12,
    session: Session | None = None,
    context: RunContext | None = None,
) -> RunResult:
    """Synchronous wrapper around :func:`arun_agent`."""
    return run_sync(
        arun_agent(
            agent,
            user_input,
            max_iterations=max_iterations,
            session=session,
            context=context,
        )
    )
