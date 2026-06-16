"""The agent loop: drive a model + tools to completion (async-first).

This is the single place that orchestrates "call model → run tools → feed
results back → repeat until done". Everything else (in-process calls, the CLI,
the A2A server) routes through here.

Concurrency: when a model requests several tools in one turn, they execute
concurrently via :func:`asyncio.gather` — including delegations to sub-agents,
so fanning out to many specialists costs one round of latency, not N.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Coroutine, TypeVar

from .messages import Message, ToolResultBlock, ToolUseBlock

if TYPE_CHECKING:  # avoid a circular import at runtime
    from .agent import Agent

_T = TypeVar("_T")


@dataclass
class Session:
    """Carries conversation history across multiple turns with an agent.

    A per-session lock serializes turns on the *same* session so concurrent
    requests sharing a session id can't corrupt its history. Distinct sessions
    run fully in parallel.
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


def run_sync(coro: Coroutine[Any, Any, _T]) -> _T:
    """Run a coroutine to completion from synchronous code.

    Uses the current thread's loop when none is running; if called from inside
    a running loop, executes the coroutine in a dedicated thread so the sync
    wrapper works anywhere (notebooks, threaded servers, nested calls).
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


async def _execute_tool_use(agent: "Agent", block: ToolUseBlock) -> ToolResultBlock:
    impl = agent.tool_map.get(block.name)
    if impl is None:
        return ToolResultBlock(block.id, f"Error: no tool named {block.name!r}", is_error=True)
    try:
        result = await impl.invoke(**block.input)
    except Exception as exc:  # tools surface failures back to the model, not the caller
        return ToolResultBlock(block.id, f"Error: {exc}", is_error=True)
    return ToolResultBlock(block.id, "" if result is None else str(result))


async def _drive(
    agent: "Agent",
    messages: list[Message],
    max_iterations: int,
) -> tuple[int, str]:
    stop_reason = "end_turn"
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        response = await agent.model.generate(
            system=agent.instructions,
            messages=messages,
            tools=agent.tools,
        )
        messages.append(response.message)
        stop_reason = response.stop_reason

        if stop_reason != "tool_use":
            return iterations, stop_reason

        # Run every requested tool in this turn concurrently.
        tool_results = await asyncio.gather(
            *(_execute_tool_use(agent, block) for block in response.message.tool_uses)
        )
        messages.append(Message(role="user", content=list(tool_results)))

    return iterations, "max_iterations"


async def arun_agent(
    agent: "Agent",
    user_input: str,
    *,
    max_iterations: int = 12,
    session: Session | None = None,
) -> RunResult:
    """Run ``agent`` on ``user_input`` until it stops calling tools (async)."""
    if session is not None:
        async with session.lock:
            messages = list(session.messages)
            messages.append(Message(role="user", content=user_input))
            iterations, stop_reason = await _drive(agent, messages, max_iterations)
            session.extend(messages)
    else:
        messages = [Message(role="user", content=user_input)]
        iterations, stop_reason = await _drive(agent, messages, max_iterations)

    output = next((m.text for m in reversed(messages) if m.role == "assistant"), "")
    return RunResult(
        output=output,
        messages=messages,
        agent=agent.name,
        iterations=iterations,
        stop_reason=stop_reason,
    )


def run_agent(
    agent: "Agent",
    user_input: str,
    *,
    max_iterations: int = 12,
    session: Session | None = None,
) -> RunResult:
    """Synchronous wrapper around :func:`arun_agent`."""
    return run_sync(
        arun_agent(agent, user_input, max_iterations=max_iterations, session=session)
    )
