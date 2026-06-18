"""Tests for the run monitor (observability data plane + dashboard)."""

from __future__ import annotations

import asyncio
import json

from synapse import Agent, Monitor, ScriptedModel, classify_tool, monitor_app, tool


def test_classify_tool_by_convention():
    assert classify_tool("ask_researcher") == "subagent"
    assert classify_tool("load_skill") == "skill"
    assert classify_tool("run_python") == "script"
    assert classify_tool("write_plan") == "plan"
    assert classify_tool("fetch_result") == "fetch"
    assert classify_tool("github.create_issue") == "mcp"
    assert classify_tool("add") == "tool"


def test_monitor_records_a_run():
    @tool
    def add(a: int, b: int) -> int:
        "Add."
        return a + b

    monitor = Monitor()
    agent = Agent("calc", model=ScriptedModel([[("add", {"a": 1, "b": 2})], "done"]), tools=[add])
    asyncio.run(agent.arun("go", hooks=monitor))

    snap = monitor.snapshot()
    assert len(snap) == 1
    run = snap[0]
    assert run["agent"] == "calc"
    assert run["status"] == "done"
    kinds = [e["kind"] for e in run["activity"]]
    assert "run_start" in kinds and "tool_start" in kinds and "run_end" in kinds
    tool_ev = next(e for e in run["activity"] if e["kind"] == "tool_start")
    assert tool_ev["name"] == "add" and tool_ev["detail"] == "tool"


def test_monitor_builds_subagent_tree():
    """A sub-agent invoked as a tool (with the monitor forwarded) nests under
    its spawning run."""
    child = Agent("researcher", model=ScriptedModel(["found it"]))
    monitor = Monitor()
    parent = Agent(
        "coordinator",
        model=ScriptedModel([[("ask_researcher", {"input": "look"})], "done"]),
        tools=[child.as_tool(hooks=monitor)],
    )
    asyncio.run(parent.arun("delegate", hooks=monitor))

    tree = monitor.tree()
    assert len(tree) == 1  # one root
    root = tree[0]
    assert root["agent"] == "coordinator"
    assert len(root["children"]) == 1
    assert root["children"][0]["agent"] == "researcher"
    # the spawn shows in the parent's activity, classified as a subagent
    spawn = next(e for e in root["activity"] if e["kind"] == "tool_start")
    assert spawn["detail"] == "subagent"


def test_concurrent_runs_stay_separate():
    monitor = Monitor()

    async def go():
        a = Agent("a", model=ScriptedModel(["one"]))
        b = Agent("b", model=ScriptedModel(["two"]))
        await asyncio.gather(a.arun("x", hooks=monitor), b.arun("y", hooks=monitor))

    asyncio.run(go())
    snap = monitor.snapshot()
    assert len(snap) == 2
    # two distinct top-level runs, neither parented to the other
    assert all(r["parent_run_id"] is None for r in snap)


def test_subscribe_receives_events():
    monitor = Monitor()

    async def go():
        q = monitor.subscribe()
        agent = Agent("calc", model=ScriptedModel(["done"]))
        await agent.arun("go", hooks=monitor)
        events = []
        while not q.empty():
            events.append(await q.get())
        return events

    events = asyncio.run(go())
    kinds = [e.kind for e in events]
    assert "run_start" in kinds and "run_end" in kinds


def test_monitor_app_serves_runs():
    monitor = Monitor()
    agent = Agent("calc", model=ScriptedModel(["done"]))
    asyncio.run(agent.arun("go", hooks=monitor))
    app = monitor_app(monitor)

    status, ctype, body = asyncio.run(_get(app, "/api/runs"))
    assert status == 200 and "application/json" in ctype
    data = json.loads(body)
    assert data["runs"][0]["agent"] == "calc"

    status, ctype, body = asyncio.run(_get(app, "/"))
    assert status == 200 and "text/html" in ctype
    assert b"run monitor" in body


async def _get(app, path):
    """Drive an ASGI GET and collect the response."""
    scope = {"type": "http", "method": "GET", "path": path, "headers": []}
    sent: list[dict] = []

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent.append(msg)

    await app(scope, receive, send)
    start = next(m for m in sent if m["type"] == "http.response.start")
    headers = {k.decode(): v.decode() for k, v in start["headers"]}
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return start["status"], headers.get("content-type", ""), body
