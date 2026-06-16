"""Tests for in-memory tracing hooks."""

from __future__ import annotations

from synapse import Agent, ScriptedModel, TracingHooks, tool


@tool
def add(a: int, b: int) -> int:
    """Add."""
    return a + b


async def test_tracing_records_run_and_tool_spans():
    tracer = TracingHooks()
    agent = Agent("a", model=ScriptedModel([[("add", {"a": 1, "b": 2})], "done"]), tools=[add])
    await agent.arun("go", hooks=tracer)

    names = [s.name for s in tracer.spans]
    assert "tool:add" in names
    assert "run:a" in names

    run_span = next(s for s in tracer.spans if s.name == "run:a")
    assert run_span.duration is not None
    assert run_span.attributes["iterations"] == 2
    assert run_span.attributes["stop_reason"] == "end_turn"

    tool_span = next(s for s in tracer.spans if s.name == "tool:add")
    assert tool_span.attributes["is_error"] is False
