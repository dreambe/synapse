"""The Model abstraction every LLM backend implements (async-first)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..messages import Message
from ..tool import Tool


@dataclass
class ModelResponse:
    """A single assistant turn returned by a backend.

    ``stop_reason`` is normalized to one of ``"end_turn"`` or ``"tool_use"``;
    the run loop only branches on whether tools were requested.
    """

    message: Message
    stop_reason: str = "end_turn"


class Model(ABC):
    """Backend interface: turn a conversation into one assistant turn.

    ``generate`` is a coroutine so many calls can be in flight concurrently —
    the framework runs a single event loop and never blocks it on network I/O.
    """

    @abstractmethod
    async def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResponse:
        """Produce the next assistant message given the conversation so far."""
        raise NotImplementedError
