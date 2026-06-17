"""Tests for the v0.7 first-principles fixes: offload, plan, idempotency,
outcome verification, run records."""

from __future__ import annotations

import re
import sys

from synapse import (
    Agent,
    InMemoryJournal,
    InMemoryResultStore,
    InMemoryRunStore,
    Plan,
    RunRecorder,
    ScriptedModel,
    command_verifier,
    tool,
)
from synapse.messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from synapse.models.base import Model, ModelResponse
from synapse.runtime import _make_fetch_tool


# -- context offloading ------------------------------------------------------


@tool
def big() -> str:
    """Return a large result."""
    return "DATA " * 1000  # ~5000 chars


async def test_large_result_is_offloaded():
    store = InMemoryResultStore()
    agent = Agent("o", model=ScriptedModel([[("big", {})], "done"]), tools=[big])
    result = await agent.arun("go", offload_over=200, result_store=store)

    tr = next(b for m in result.messages for b in m.content if isinstance(b, ToolResultBlock))
    assert isinstance(tr.content, str) and tr.content.startswith("[Large result offloaded")
    assert len(tr.content) < 1000  # the 5000-char result is no longer in context
    ref = re.search(r"res_\w+", tr.content).group()
    assert (await store.get(ref)).startswith("DATA")


async def test_fetch_result_tool_reads_and_filters():
    store = InMemoryResultStore()
    ref = await store.put("alpha line\nbeta line\ngamma line")
    fetch = _make_fetch_tool(store)
    assert "alpha" in await fetch.invoke(ref=ref)
    assert await fetch.invoke(ref=ref, contains="beta") == "beta line"
    assert "no offloaded result" in await fetch.invoke(ref="res_nope")


# -- explicit plan -----------------------------------------------------------


class PlanCaptureModel(Model):
    """Writes a plan, marks a step done, then finishes — recording the system
    prompt each turn so we can prove the plan is injected as context."""

    def __init__(self) -> None:
        self.systems: list[str] = []
        self.turn = 0

    async def generate(self, *, system, messages, tools):
        self.systems.append(system or "")
        self.turn += 1
        if self.turn == 1:
            return ModelResponse(
                Message("assistant", [ToolUseBlock("c1", "write_plan", {"steps": ["scope", "build"]})]),
                "tool_use",
            )
        if self.turn == 2:
            return ModelResponse(
                Message("assistant", [ToolUseBlock("c2", "update_step", {"step": "scope", "status": "done"})]),
                "tool_use",
            )
        return ModelResponse(Message("assistant", [TextBlock("done")]))


async def test_plan_is_maintained_and_injected():
    model = PlanCaptureModel()
    agent = Agent("p", instructions="Work.", model=model)
    result = await agent.arun("do a thing", plan=True)

    assert result.plan is not None
    statuses = {s.text: s.status for s in result.plan.steps}
    assert statuses == {"scope": "done", "build": "pending"}
    # by turn 2 the plan (set in turn 1) is in the system prompt
    assert "# Plan" in model.systems[1] and "scope" in model.systems[1]


def test_plan_render():
    from synapse.plan import PlanStep

    plan = Plan([PlanStep("a", "done"), PlanStep("b", "in_progress"), PlanStep("c")])
    rendered = plan.render()
    assert "[x] a" in rendered and "[~] b" in rendered and "[ ] c" in rendered


# -- idempotent execution (journal) -----------------------------------------


class ActThenDone(Model):
    async def generate(self, *, system, messages, tools):
        has_result = any(
            getattr(b, "type", None) == "tool_result" for m in messages for b in m.content
        )
        if has_result:
            return ModelResponse(Message("assistant", [TextBlock("done")]))
        return ModelResponse(Message("assistant", [ToolUseBlock("c", "act", {})]), "tool_use")


async def test_journal_replays_instead_of_re_executing():
    calls: list[int] = []

    @tool
    def act() -> str:
        """A side-effecting action."""
        calls.append(1)
        return "did it"

    journal = InMemoryJournal()
    agent = Agent("j", model=ActThenDone(), tools=[act])

    await agent.arun("go", journal=journal, run_id="run-1")
    assert calls == [1]  # executed once

    # Replay the same run id + same call sequence → tool is NOT re-executed.
    await agent.arun("go", journal=journal, run_id="run-1")
    assert calls == [1]  # still once — the side effect did not fire again


async def test_journal_off_by_default_re_executes():
    calls: list[int] = []

    @tool
    def act() -> str:
        """Action."""
        calls.append(1)
        return "did"

    agent = Agent("j", model=ActThenDone(), tools=[act])
    await agent.arun("go")
    await agent.arun("go")  # no journal → runs each time
    assert calls == [1, 1]


# -- outcome verification ----------------------------------------------------


async def test_command_verifier_pass():
    agent = Agent("v", model=ScriptedModel(["built it"]))
    result = await agent.arun(
        "build", verify=command_verifier([sys.executable, "-c", "import sys; sys.exit(0)"])
    )
    assert result.stop_reason == "verified"


async def test_command_verifier_fail_feeds_back():
    agent = Agent("v", model=ScriptedModel(["attempt 1", "attempt 2"]))
    result = await agent.arun(
        "build",
        verify=command_verifier([sys.executable, "-c", "import sys; sys.exit(1)"]),
        max_verify_rounds=2,
    )
    assert result.stop_reason == "verification_failed"
    # the failing check's output was fed back as a revise message
    assert any("rejected by verification" in m.text for m in result.messages if m.role == "user")


# -- run records (fact source) ----------------------------------------------


async def test_run_recorder_persists_a_record():
    store = InMemoryRunStore()
    agent = Agent("r", model=ScriptedModel([[("act", {})], "finished"]), tools=[])
    await agent.arun("do the task", hooks=RunRecorder(store))

    records = await store.list()
    assert len(records) == 1
    rec = records[0]
    assert rec.agent == "r"
    assert rec.input == "do the task"
    assert rec.output == "finished"
    assert rec.stop_reason in ("end_turn", "max_iterations")
    assert rec.messages  # the full trajectory is captured
    assert await store.get(rec.run_id) is not None
