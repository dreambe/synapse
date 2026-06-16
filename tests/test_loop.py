"""Loop engineering: stopping guards beyond the iteration cap."""

from __future__ import annotations

from synapse import Agent, tool
from synapse.messages import Message, ToolUseBlock
from synapse.models.base import Model, ModelResponse


class RepeatToolModel(Model):
    """Always requests the same tool call — never stops on its own."""

    def __init__(self, name: str, tool_input: dict | None = None) -> None:
        self.name = name
        self.tool_input = tool_input or {}

    async def generate(self, *, system, messages, tools):
        block = ToolUseBlock(id="c", name=self.name, input=self.tool_input)
        return ModelResponse(Message("assistant", [block]), stop_reason="tool_use")


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


@tool
def boom() -> str:
    """Always fails."""
    raise RuntimeError("kaboom")


async def test_loop_detection_stops_repeated_calls():
    agent = Agent("l", model=RepeatToolModel("noop"), tools=[noop])
    result = await agent.arun("go", max_repeated_tool_calls=2, max_iterations=20)
    assert result.stop_reason == "loop_detected"
    assert result.iterations <= 4  # tripped quickly, not at the iteration cap


async def test_no_progress_detection():
    agent = Agent("l", model=RepeatToolModel("noop"), tools=[noop])
    result = await agent.arun("go", max_no_progress=2, max_iterations=20)
    assert result.stop_reason == "no_progress"
    assert result.iterations <= 4


async def test_circuit_breaker_on_consecutive_tool_errors():
    agent = Agent("l", model=RepeatToolModel("boom"), tools=[boom])
    result = await agent.arun("go", max_consecutive_tool_errors=2, max_iterations=20)
    assert result.stop_reason == "tool_errors_exhausted"
    assert result.iterations <= 4


async def test_guards_off_by_default_fall_back_to_iteration_cap():
    agent = Agent("l", model=RepeatToolModel("noop"), tools=[noop])
    result = await agent.arun("go", max_iterations=3)
    # No loop guards set → only the hard iteration cap stops it.
    assert result.stop_reason == "max_iterations"
    assert result.iterations == 3


async def test_parallel_identical_calls_not_falsely_flagged():
    # Four identical calls in ONE turn (legit fan-out) must not trip loop
    # detection, which counts per-turn occurrences, not per-call.
    from synapse import ScriptedModel

    agent = Agent(
        "l",
        model=ScriptedModel([[("noop", {}), ("noop", {}), ("noop", {}), ("noop", {})], "done"]),
        tools=[noop],
    )
    result = await agent.arun("go", max_repeated_tool_calls=2)
    assert result.output == "done"
    assert result.stop_reason == "end_turn"
