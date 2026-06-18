"""Tests for durable execution: atomic mid-batch recovery on resume.

A run dies after part of a parallel tool batch has executed. On resume, the
completed tools must NOT re-fire their side effects, the unfinished ones must
run, and the batch must complete so the run can proceed.
"""

from __future__ import annotations

import asyncio

from synapse import (
    Agent,
    InMemoryCheckpointer,
    InMemoryJournal,
    ScriptedModel,
    tool,
)
from synapse.journal import call_key
from synapse.messages import Message, ToolUseBlock


def test_mid_batch_recovery_replays_done_skips_refire():
    fired: list[str] = []

    @tool
    def charge_a() -> str:
        "Side effect A."
        fired.append("a")
        return "charged A"

    @tool
    def charge_b() -> str:
        "Side effect B."
        fired.append("b")
        return "charged B"

    run_id = "run-x"
    journal = InMemoryJournal()
    checkpointer = InMemoryCheckpointer()

    # Simulate a crash: the assistant asked for a 2-tool batch; only charge_a
    # finished (recorded in the journal) before the process died. The checkpoint
    # holds the transcript up to the unfulfilled tool-use turn.
    batch = [
        ToolUseBlock(id="t0", name="charge_a", input={}),
        ToolUseBlock(id="t1", name="charge_b", input={}),
    ]
    crashed = [
        Message(role="user", content="please charge both"),
        Message(role="assistant", content=batch),
    ]
    asyncio.run(checkpointer.save(run_id, crashed))
    asyncio.run(
        journal.record(
            call_key(run_id, "charge_a", {}, 0), {"content": "charged A", "is_error": False}
        )
    )

    agent = Agent("pay", model=ScriptedModel(["all done"]), tools=[charge_a, charge_b])
    res = asyncio.run(
        agent.arun("please charge both", run_id=run_id, journal=journal, checkpointer=checkpointer)
    )

    # charge_a replayed from the journal (no re-fire); charge_b actually ran.
    assert fired == ["b"]
    assert res.output == "all done"
    # Both tool results are present in the transcript.
    from synapse import ToolResultBlock

    contents = [
        b.content
        for m in res.messages
        if isinstance(m.content, list)
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert "charged A" in contents and "charged B" in contents


def test_fresh_run_is_unaffected():
    @tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    agent = Agent("calc", model=ScriptedModel([[("add", {"a": 2, "b": 3})], "5"]), tools=[add])
    res = asyncio.run(
        agent.arun("2+3", run_id="r2", journal=InMemoryJournal(), checkpointer=InMemoryCheckpointer())
    )
    assert res.output == "5"
