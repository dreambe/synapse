"""Tests for mid-run steering (inject guidance / graceful stop)."""

from __future__ import annotations

import asyncio

from synapse import Agent, ScriptedModel, Steer, Steered, tool


def test_stop_request_halts_gracefully():
    steer = Steer()
    steer.stop("that's enough")
    agent = Agent("a", model=ScriptedModel(["one", "two", "three"]))
    res = asyncio.run(agent.arun("go", steer=steer))
    assert res.stop_reason == "steered_stop"
    assert steer.stop_feedback == "that's enough"


def test_injected_guidance_enters_the_transcript_mid_run():
    """A tool fires a steer; the next turn boundary injects it as a user turn."""
    steer = Steer()

    @tool
    def trigger() -> str:
        "Fires a steer mid-run."
        steer.send("focus on X")
        return "ok"

    agent = Agent("a", model=ScriptedModel([[("trigger", {})], "done"]), tools=[trigger])
    res = asyncio.run(agent.arun("go", steer=steer))
    injected = [m for m in res.messages if m.role == "user" and m.text == "[steering] focus on X"]
    assert len(injected) == 1
    assert res.output == "done"


def test_streaming_emits_steered_event():
    steer = Steer()
    steer.send("hello there")  # drained at the first turn boundary

    async def go():
        agent = Agent("a", model=ScriptedModel(["done"]))
        events = []
        async for ev in agent.astream("go", steer=steer):
            events.append(ev)
        return events

    events = asyncio.run(go())
    steered = [e for e in events if isinstance(e, Steered)]
    assert len(steered) == 1 and steered[0].text == "hello there"


def test_concurrent_steer_from_another_task():
    """The realistic shape: run in one task, steer from another."""
    steer = Steer()
    gate = asyncio.Event()

    @tool
    async def slow() -> str:
        "Lets the supervisor steer before the next turn."
        gate.set()
        await asyncio.sleep(0.01)
        return "ok"

    async def go():
        agent = Agent("a", model=ScriptedModel([[("slow", {})], "done"]), tools=[slow])
        task = asyncio.create_task(agent.arun("go", steer=steer))
        await gate.wait()
        steer.send("steered while running")
        return await task

    res = asyncio.run(go())
    assert any(m.text == "[steering] steered while running" for m in res.messages if m.role == "user")


def test_drain_is_idempotent():
    steer = Steer()
    steer.send("a")
    steer.send("b")
    assert steer.pending == 2
    assert steer.drain() == ["a", "b"]
    assert steer.drain() == [] and steer.pending == 0
