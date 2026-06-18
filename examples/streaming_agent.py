"""Stream a run event-by-event — text deltas, tool calls, and the final result.

Offline demo (no API key): a backend that emits word-by-word deltas.

    python examples/streaming_agent.py
"""

from __future__ import annotations

import asyncio

from synapse import Agent
from synapse.messages import Message, TextBlock
from synapse.models.base import Model, ModelResponse
from synapse.streaming import ModelStreamEnd, TextDelta


class WordStreamModel(Model):
    """A backend with native streaming: yields one delta per word."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(self, *, system, messages, tools):
        return ModelResponse(Message("assistant", [TextBlock(self.text)]))

    async def stream(self, *, system, messages, tools):
        for word in self.text.split():
            await asyncio.sleep(0.05)
            yield TextDelta(word + " ")
        yield ModelStreamEnd(ModelResponse(Message("assistant", [TextBlock(self.text)])))


async def main() -> None:
    agent = Agent("poet", model=WordStreamModel("synapse streams events as they happen"))
    async for event in agent.astream("write something"):
        if event.type == "text_delta":
            print(event.text, end="", flush=True)
        elif event.type == "run_complete":
            print(f"\n[done in {event.result.iterations} turn(s)]")


if __name__ == "__main__":
    asyncio.run(main())
