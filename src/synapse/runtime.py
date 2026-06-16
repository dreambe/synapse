"""The agent loop.

Streaming is the primitive: :func:`arun_stream` drives the model/tool loop and
yields events; the non-streaming :func:`arun_agent` simply consumes that stream
to its final :class:`RunResult`. One loop, one source of truth.

Capabilities are threaded through a single :class:`RunContext` (the canonical
configuration object — the keyword arguments on :meth:`Agent.run` are sugar
over it): observability hooks, tool approval (HITL), guardrails, a verifier
(iterate-until-pass), token budgets, wall-clock + per-tool timeouts, bounded
tool concurrency, context compaction, checkpointing, and tool search.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Optional,
    TypeVar,
    Union,
)

from ._util import maybe_await
from .errors import SynapseError
from .guardrails import Guardrail, apply_guardrails
from .messages import ImageBlock, Message, TextBlock, ToolResultBlock, ToolUseBlock
from .observability import Hooks, Usage
from .streaming import ModelStreamEnd, RunComplete, RunEvent, TextDelta, ToolCall, ToolOutput
from .tool import Tool

if TYPE_CHECKING:  # avoid circular imports at runtime
    from .agent import Agent
    from .checkpoint import Checkpointer
    from .context import Compactor

_T = TypeVar("_T")


class RunTimeout(SynapseError):
    """Raised when a run exceeds its wall-clock ``timeout``."""


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


ApprovalCallback = Callable[[str, dict], Union[bool, ApprovalDecision, Awaitable[Any]]]
Verifier = Callable[[str], Union[bool, Verdict, Awaitable[Any]]]


@dataclass
class Session:
    """Conversation history across turns. A per-session lock serializes turns
    on the same session; distinct sessions run fully in parallel."""

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
    """The canonical run configuration. All capabilities live here."""

    max_iterations: int = 12
    hooks: Optional[Hooks] = None
    approval: Optional[ApprovalCallback] = None
    verify: Optional[Verifier] = None
    max_verify_rounds: int = 3
    token_budget: Optional[int] = None
    timeout: Optional[float] = None
    tool_timeout: Optional[float] = None
    max_parallel_tools: Optional[int] = None
    input_guardrails: list[Guardrail] = field(default_factory=list)
    output_guardrails: list[Guardrail] = field(default_factory=list)
    compactor: Optional["Compactor"] = None
    checkpointer: Optional["Checkpointer"] = None
    run_id: Optional[str] = None
    tool_search: bool = False
    usage: Usage = field(default_factory=Usage)


@dataclass
class _TurnInfo:
    """Internal sentinel ending a drive stream."""

    iterations: int
    stop_reason: str
    verify_rounds: int = 1


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from synchronous code, anywhere.

    Caveat: when called from *inside* a running event loop this spins up a
    separate loop in a worker thread, so resources bound to the parent loop
    (clients, pools) are not shared. Prefer the async API in async code.
    """
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
    return value if isinstance(value, ApprovalDecision) else ApprovalDecision(allow=bool(value))


_BLOCK_TYPES = (TextBlock, ImageBlock, ToolUseBlock, ToolResultBlock)


def _tool_result_content(result: Any):
    """Coerce a tool's return value into tool_result content.

    Rich returns (an ``ImageBlock``, or a list of content blocks) pass through
    for multimodal results; everything else is stringified.
    """
    if result is None:
        return ""
    if isinstance(result, ImageBlock):
        return [result]
    if isinstance(result, (list, tuple)) and all(isinstance(b, _BLOCK_TYPES) for b in result):
        return list(result)
    if isinstance(result, str):
        return result
    return str(result)


def _content_text(content: Any) -> str:
    """A short textual rendering of (possibly multimodal) content for events."""
    if isinstance(content, str):
        return content
    parts = []
    for b in content:
        parts.append(b.text if isinstance(b, TextBlock) else f"[{getattr(b, 'type', 'block')}]")
    return " ".join(parts)


def _render_input(user_input: Any) -> str:
    if isinstance(user_input, str):
        return user_input
    return " ".join(b.text for b in user_input if isinstance(b, TextBlock))


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
    sem: Optional[asyncio.Semaphore],
) -> ToolResultBlock:
    impl = tool_map.get(block.name)
    if impl is None:
        return ToolResultBlock(block.id, f"Error: no tool named {block.name!r}", is_error=True)

    if impl.requires_approval and ctx.approval is not None:
        decision = _normalize_approval(await maybe_await(ctx.approval(block.name, block.input)))
        if not decision.allow:
            reason = decision.reason or "denied by approval policy"
            return ToolResultBlock(block.id, f"Tool call denied: {reason}", is_error=True)

    await _emit(ctx, "on_tool_start", block.name, block.input)
    guard = sem if sem is not None else contextlib.nullcontext()
    try:
        async with guard:
            call = impl.invoke(**block.input)
            if ctx.tool_timeout is not None:
                result = await asyncio.wait_for(call, ctx.tool_timeout)
            else:
                result = await call
        out = ToolResultBlock(block.id, _tool_result_content(result))
    except asyncio.TimeoutError:
        out = ToolResultBlock(
            block.id, f"Error: tool {block.name!r} timed out after {ctx.tool_timeout}s", is_error=True
        )
    except Exception as exc:  # tools surface failures back to the model
        out = ToolResultBlock(block.id, f"Error ({type(exc).__name__}): {exc}", is_error=True)
    await _emit(ctx, "on_tool_end", block.name, out.content, out.is_error)
    return out


async def _drive_stream(
    ctx: RunContext,
    agent: "Agent",
    messages: list[Message],
    deadline: Optional[float],
) -> AsyncIterator[Union[RunEvent, _TurnInfo]]:
    active: set[str] = set()
    base_map = agent.tool_map
    search_tool = _make_search_tool(agent, active) if ctx.tool_search else None
    sem = asyncio.Semaphore(ctx.max_parallel_tools) if ctx.max_parallel_tools else None
    loop = asyncio.get_running_loop()

    def exposed_tools() -> list[Tool]:
        if search_tool is None:
            return agent.tools
        return [search_tool, *[t for t in agent.tools if t.name in active]]

    def tool_map() -> dict[str, Tool]:
        if search_tool is None:
            return base_map
        return {search_tool.name: search_tool, **base_map}

    iterations = 0
    for iterations in range(1, ctx.max_iterations + 1):
        if deadline is not None and loop.time() > deadline:
            raise RunTimeout(f"run exceeded {ctx.timeout}s")

        if ctx.compactor is not None:
            messages[:] = await ctx.compactor.maybe_compact(messages)

        response = None
        async for chunk in agent.model.stream(
            system=agent.instructions, messages=messages, tools=exposed_tools()
        ):
            if isinstance(chunk, TextDelta):
                yield chunk
            elif isinstance(chunk, ModelStreamEnd):
                response = chunk.response
        assert response is not None, "model stream ended without a final message"

        messages.append(response.message)
        ctx.usage.add(response.usage)
        await _emit(ctx, "on_model_response", response)
        await _emit(ctx, "on_turn_end", iterations, response.message)

        if ctx.checkpointer is not None and ctx.run_id is not None:
            await ctx.checkpointer.save(ctx.run_id, messages)

        if response.stop_reason != "tool_use":
            yield _TurnInfo(iterations, response.stop_reason)
            return

        if ctx.token_budget is not None and ctx.usage.total_tokens >= ctx.token_budget:
            yield _TurnInfo(iterations, "budget_exceeded")
            return

        uses = response.message.tool_uses
        for block in uses:
            yield ToolCall(id=block.id, name=block.name, input=block.input)

        results = await asyncio.gather(
            *(_execute_tool_use(ctx, tool_map(), block, sem) for block in uses)
        )
        for r in results:
            yield ToolOutput(
                id=r.tool_use_id, name="", content=_content_text(r.content), is_error=r.is_error
            )
        messages.append(Message(role="user", content=list(results)))

        if ctx.checkpointer is not None and ctx.run_id is not None:
            await ctx.checkpointer.save(ctx.run_id, messages)

    yield _TurnInfo(iterations, "max_iterations")


async def _verify_stream(
    ctx: RunContext,
    agent: "Agent",
    messages: list[Message],
    deadline: Optional[float],
) -> AsyncIterator[Union[RunEvent, _TurnInfo]]:
    total = 0
    stop_reason = "end_turn"
    rounds = 0
    for rounds in range(1, ctx.max_verify_rounds + 1):
        async for ev in _drive_stream(ctx, agent, messages, deadline):
            if isinstance(ev, _TurnInfo):
                total += ev.iterations
                stop_reason = ev.stop_reason
            else:
                yield ev
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
    yield _TurnInfo(total, stop_reason, verify_rounds=rounds)


async def arun_stream(
    agent: "Agent",
    user_input: str,
    *,
    session: Session | None = None,
    context: RunContext | None = None,
) -> AsyncIterator[RunEvent]:
    """Run ``agent`` and yield events as they happen, ending with
    :class:`~synapse.streaming.RunComplete`."""
    ctx = context or RunContext()
    loop = asyncio.get_running_loop()
    deadline = loop.time() + ctx.timeout if ctx.timeout else None

    await _emit(ctx, "on_run_start", agent.name, _render_input(user_input))
    if isinstance(user_input, str):
        user_content: Any = await apply_guardrails(user_input, ctx.input_guardrails)
    else:
        # Multimodal input (list of blocks): guardrails run on text only.
        user_content = user_input

    async def _stream_body() -> AsyncIterator[RunEvent]:
        messages: list[Message] = []
        if session is not None:
            messages = list(session.messages)
        elif ctx.checkpointer is not None and ctx.run_id is not None:
            restored = await ctx.checkpointer.load(ctx.run_id)
            if restored:
                messages = restored
        messages.append(Message(role="user", content=user_content))

        iterations = 0
        stop_reason = "end_turn"
        rounds = 1
        async for ev in _verify_stream(ctx, agent, messages, deadline):
            if isinstance(ev, _TurnInfo):
                iterations, stop_reason, rounds = ev.iterations, ev.stop_reason, ev.verify_rounds
            else:
                yield ev

        if session is not None:
            session.extend(messages)

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
        yield RunComplete(result=result)

    if session is not None:
        async with session.lock:
            async for ev in _stream_body():
                yield ev
    else:
        async for ev in _stream_body():
            yield ev


async def arun_agent(
    agent: "Agent",
    user_input: str,
    *,
    max_iterations: int = 12,
    session: Session | None = None,
    context: RunContext | None = None,
) -> RunResult:
    """Run ``agent`` on ``user_input`` until done (async); returns the result."""
    ctx = context or RunContext(max_iterations=max_iterations)
    if context is None:
        ctx.max_iterations = max_iterations

    async def _collect() -> RunResult:
        result: RunResult | None = None
        async for ev in arun_stream(agent, user_input, session=session, context=ctx):
            if isinstance(ev, RunComplete):
                result = ev.result
        assert result is not None
        return result

    if ctx.timeout is not None:
        try:
            return await asyncio.wait_for(_collect(), ctx.timeout)
        except asyncio.TimeoutError as exc:
            raise RunTimeout(f"run exceeded {ctx.timeout}s") from exc
    return await _collect()


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
            agent, user_input, max_iterations=max_iterations, session=session, context=context
        )
    )
