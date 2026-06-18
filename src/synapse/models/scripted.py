"""Backends that need no API key — for tests, demos, and offline development.

:class:`EchoModel` parrots the last user message. :class:`ScriptedModel`
replays a fixed sequence of turns (including tool calls), which makes the run
loop and A2A transport fully testable without a network.
"""

from __future__ import annotations

import asyncio
from typing import Union

from ..messages import Message, TextBlock, ToolUseBlock
from .base import Model, ModelResponse

# A scripted step is either final text, or a list of (tool_name, input) calls.
Step = Union[str, list[tuple[str, dict]]]


class EchoModel(Model):
    """Repeats the most recent user text back. Never calls tools."""

    def __init__(self, prefix: str = "echo: ") -> None:
        self.prefix = prefix

    async def generate(self, *, system, messages, tools) -> ModelResponse:
        last_user = next(
            (m.text for m in reversed(messages) if m.role == "user" and m.text),
            "",
        )
        return ModelResponse(Message("assistant", [TextBlock(self.prefix + last_user)]))


class ScriptedModel(Model):
    """Replays a predetermined list of turns.

    Each step is either a string (a final text answer) or a list of
    ``(tool_name, input_dict)`` tuples (a tool-use turn). The recorded
    ``conversations`` let tests assert on what the loop fed the model.

    Access to the script is guarded by a lock so the same instance can be
    shared across concurrent runs without losing or interleaving steps.
    """

    def __init__(self, script: list[Step]) -> None:
        self.script = list(script)
        self.conversations: list[list[Message]] = []
        self._lock = asyncio.Lock()

    async def generate(self, *, system, messages, tools) -> ModelResponse:
        async with self._lock:
            self.conversations.append(list(messages))
            if not self.script:
                return ModelResponse(Message("assistant", [TextBlock("(script exhausted)")]))
            step = self.script.pop(0)
        if isinstance(step, str):
            return ModelResponse(Message("assistant", [TextBlock(step)]))
        blocks = [
            ToolUseBlock(id=f"call_{i}", name=name, input=inp)
            for i, (name, inp) in enumerate(step)
        ]
        return ModelResponse(Message("assistant", blocks), stop_reason="tool_use")
