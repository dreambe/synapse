"""Observability: token usage, lifecycle events, and hooks.

Tracing is treated as framework infrastructure, not an afterthought — every
run can emit structured events that feed monitoring, evaluation, and
debugging. Hooks may be sync or async.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ._util import maybe_await

if TYPE_CHECKING:
    from .messages import Message
    from .models.base import ModelResponse
    from .runtime import RunResult


@dataclass
class Usage:
    """Token accounting for a run (summed across model calls)."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
        )

    def add(self, other: "Usage | None") -> None:
        if other is not None:
            self.input_tokens += other.input_tokens
            self.output_tokens += other.output_tokens


class Hooks:
    """Lifecycle callbacks for a run. Subclass and override what you need.

    Every method is a no-op by default and may be implemented as sync or async.
    """

    async def on_run_start(self, agent: str, user_input: str) -> None: ...
    async def on_model_response(self, response: "ModelResponse") -> None: ...
    async def on_tool_start(self, name: str, tool_input: dict) -> None: ...
    async def on_tool_end(self, name: str, result: str, is_error: bool) -> None: ...
    async def on_turn_end(self, index: int, message: "Message") -> None: ...
    async def on_run_end(self, result: "RunResult") -> None: ...


class CollectingHooks(Hooks):
    """Records every event into ``events`` — handy for tests and inspection."""

    def __init__(self) -> None:
        self.events: list[tuple[str, dict]] = []

    async def on_run_start(self, agent: str, user_input: str) -> None:
        self.events.append(("run_start", {"agent": agent, "input": user_input}))

    async def on_model_response(self, response: "ModelResponse") -> None:
        self.events.append(("model_response", {"stop_reason": response.stop_reason}))

    async def on_tool_start(self, name: str, tool_input: dict) -> None:
        self.events.append(("tool_start", {"name": name, "input": tool_input}))

    async def on_tool_end(self, name: str, result: str, is_error: bool) -> None:
        self.events.append(("tool_end", {"name": name, "is_error": is_error}))

    async def on_turn_end(self, index: int, message: "Message") -> None:
        self.events.append(("turn_end", {"index": index, "role": message.role}))

    async def on_run_end(self, result: "RunResult") -> None:
        self.events.append(("run_end", {"output": result.output}))


class CompositeHooks(Hooks):
    """Fan an event out to several hook objects."""

    def __init__(self, *hooks: Hooks) -> None:
        self._hooks = list(hooks)

    async def _broadcast(self, method: str, *args: Any) -> None:
        for hook in self._hooks:
            await maybe_await(getattr(hook, method)(*args))

    async def on_run_start(self, agent: str, user_input: str) -> None:
        await self._broadcast("on_run_start", agent, user_input)

    async def on_model_response(self, response: "ModelResponse") -> None:
        await self._broadcast("on_model_response", response)

    async def on_tool_start(self, name: str, tool_input: dict) -> None:
        await self._broadcast("on_tool_start", name, tool_input)

    async def on_tool_end(self, name: str, result: str, is_error: bool) -> None:
        await self._broadcast("on_tool_end", name, result, is_error)

    async def on_turn_end(self, index: int, message: "Message") -> None:
        await self._broadcast("on_turn_end", index, message)

    async def on_run_end(self, result: "RunResult") -> None:
        await self._broadcast("on_run_end", result)
