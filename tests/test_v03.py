"""Tests for v0.3: streaming, timeouts, and bounded tool concurrency."""

from __future__ import annotations

import asyncio

import pytest

from synapse import (
    Agent,
    RunComplete,
    RunContext,
    RunTimeout,
    ScriptedModel,
    TextDelta,
    tool,
)
from synapse.messages import Message, TextBlock
from synapse.models.base import Model, ModelResponse
from synapse.streaming import ModelStreamEnd


class DeltaModel(Model):
    """A backend with native streaming: emits words as separate deltas."""

    def __init__(self, text: str) -> None:
        self.text = text

    async def generate(self, *, system, messages, tools):
        return ModelResponse(Message("assistant", [TextBlock(self.text)]))

    async def stream(self, *, system, messages, tools):
        for word in self.text.split():
            yield TextDelta(word + " ")
        yield ModelStreamEnd(
            ModelResponse(Message("assistant", [TextBlock(self.text)]))
        )


@tool
def add(a: int, b: int) -> int:
    """Add."""
    return a + b


async def test_stream_yields_text_deltas_then_complete():
    agent = Agent("s", model=DeltaModel("hello there world"))
    events = [ev async for ev in agent.astream("hi")]
    deltas = [e.text for e in events if isinstance(e, TextDelta)]
    assert "".join(deltas).split() == ["hello", "there", "world"]
    assert isinstance(events[-1], RunComplete)
    assert events[-1].result.output == "hello there world"


async def test_stream_emits_tool_events():
    agent = Agent(
        "s",
        model=ScriptedModel([[("add", {"a": 2, "b": 3})], "done"]),
        tools=[add],
    )
    kinds = [type(ev).__name__ async for ev in agent.astream("go")]
    assert "ToolCall" in kinds and "ToolOutput" in kinds
    assert kinds[-1] == "RunComplete"


async def test_non_stream_consumes_stream():
    # The sync/async result path must agree with streaming.
    agent = Agent("s", model=DeltaModel("final answer"))
    result = await agent.arun("hi")
    assert result.output == "final answer"


# -- timeouts ---------------------------------------------------------------


class SlowModel(Model):
    def __init__(self, delay: float) -> None:
        self.delay = delay

    async def generate(self, *, system, messages, tools):
        await asyncio.sleep(self.delay)
        return ModelResponse(Message("assistant", [TextBlock("late")]))


async def test_run_timeout_raises():
    agent = Agent("t", model=SlowModel(0.5))
    with pytest.raises(RunTimeout):
        await agent.arun("go", timeout=0.1)


async def test_tool_timeout_surfaces_as_error():
    @tool
    async def slow_tool() -> str:
        """Sleeps too long."""
        await asyncio.sleep(0.5)
        return "done"

    agent = Agent(
        "t",
        model=ScriptedModel([[("slow_tool", {})], "recovered"]),
        tools=[slow_tool],
    )
    result = await agent.arun("go", tool_timeout=0.1)
    assert result.output == "recovered"
    errors = [
        b
        for m in result.messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result" and b.is_error
    ]
    assert errors and "timed out" in errors[0].content


# -- bounded concurrency ----------------------------------------------------


async def test_max_parallel_tools_bounds_concurrency():
    active = {"now": 0, "peak": 0}

    @tool
    async def track() -> str:
        """Tracks concurrent executions."""
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.05)
        active["now"] -= 1
        return "ok"

    # One turn requesting the tool four times in parallel, capped at 2.
    agent = Agent(
        "c",
        model=ScriptedModel([[("track", {}), ("track", {}), ("track", {}), ("track", {})], "done"]),
        tools=[track],
    )
    await agent.arun("go", max_parallel_tools=2)
    assert active["peak"] <= 2


async def test_unbounded_runs_all_in_parallel():
    active = {"now": 0, "peak": 0}

    @tool
    async def track() -> str:
        """Tracks concurrent executions."""
        active["now"] += 1
        active["peak"] = max(active["peak"], active["now"])
        await asyncio.sleep(0.05)
        active["now"] -= 1
        return "ok"

    agent = Agent(
        "c",
        model=ScriptedModel([[("track", {}), ("track", {}), ("track", {})], "done"]),
        tools=[track],
    )
    await agent.arun("go")  # no cap → all three at once
    assert active["peak"] == 3


def test_runcontext_is_canonical():
    # Passing a RunContext directly is equivalent to kwargs.
    agent = Agent("c", model=ScriptedModel(["ok"]))
    result = agent.run("hi", context=RunContext(max_iterations=5))
    assert result.output == "ok"
