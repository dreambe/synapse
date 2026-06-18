"""The Model abstraction every LLM backend implements (async-first)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator

from ..messages import Message
from ..observability import Usage
from ..streaming import ModelChunk, ModelStreamEnd, TextDelta
from ..tool import Tool


@dataclass
class ModelResponse:
    """A single assistant turn returned by a backend.

    ``stop_reason`` is normalized to one of ``"end_turn"`` or ``"tool_use"``;
    the run loop only branches on whether tools were requested. ``usage``
    carries token accounting when the backend reports it.
    """

    message: Message
    stop_reason: str = "end_turn"
    usage: Usage | None = None


class Model(ABC):
    """Backend interface: turn a conversation into one assistant turn.

    ``generate`` is the required primitive. ``stream`` yields incremental
    chunks; the default implementation derives a (single-chunk) stream from
    ``generate`` so every backend streams for free — override it for real
    token-level streaming.
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

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
    ) -> AsyncIterator[ModelChunk]:
        """Yield text deltas, then a single :class:`ModelStreamEnd`.

        Default: call :meth:`generate` and emit its text as one delta. Backends
        with native streaming should override this.
        """
        response = await self.generate(system=system, messages=messages, tools=tools)
        text = response.message.text
        if text:
            yield TextDelta(text)
        yield ModelStreamEnd(response)
