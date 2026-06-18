"""Tests for structured permission modes."""

from __future__ import annotations

import asyncio

from synapse import (
    Agent,
    ApprovalDecision,
    PermissionMode,
    ScriptedModel,
    permission_policy,
    tool,
)


@tool(side_effect="read")
def look(x: int) -> str:
    "A read-only tool."
    return f"saw {x}"


@tool  # default side_effect="write"
def mutate(x: int) -> str:
    "A mutating tool."
    return f"changed {x}"


def _agent(script):
    return Agent("a", model=ScriptedModel(script), tools=[look, mutate])


def test_plan_mode_denies_writes_allows_reads():
    agent = _agent([[("look", {"x": 1}), ("mutate", {"x": 2})], "done"])
    res = asyncio.run(agent.arun("go", permissions=permission_policy(PermissionMode.PLAN)))
    results = _tool_results(res)
    assert any("saw 1" in r for r in results)
    assert any("denied" in r.lower() and "mutate" in r for r in results)


def test_ask_mode_escalates_writes_to_approval():
    seen = []

    def approve(name, inp):
        seen.append(name)
        return ApprovalDecision(allow=False, reason="not allowed")

    agent = _agent([[("mutate", {"x": 1})], "done"])
    res = asyncio.run(
        agent.arun("go", permissions=permission_policy("ask"), approval=approve)
    )
    assert seen == ["mutate"]
    assert any("denied" in r.lower() for r in _tool_results(res))


def test_ask_mode_without_approver_denies():
    agent = _agent([[("mutate", {"x": 1})], "done"])
    res = asyncio.run(agent.arun("go", permissions=permission_policy("ask")))
    assert any("no approver" in r.lower() for r in _tool_results(res))


def test_auto_mode_allows_everything():
    agent = _agent([[("mutate", {"x": 9})], "done"])
    res = asyncio.run(agent.arun("go", permissions=permission_policy("auto")))
    assert any("changed 9" in r for r in _tool_results(res))


def test_explicit_deny_and_allow_win_over_mode():
    # deny a read in PLAN; allow a write in PLAN
    agent = _agent([[("look", {"x": 1}), ("mutate", {"x": 2})], "done"])
    pol = permission_policy(PermissionMode.PLAN, allow={"mutate"}, deny={"look"})
    res = asyncio.run(agent.arun("go", permissions=pol))
    results = _tool_results(res)
    assert any("changed 2" in r for r in results)  # write allowed by exception
    assert any("denied" in r.lower() and "look" in r for r in results)  # read denied


def _tool_results(res):
    from synapse import ToolResultBlock

    out = []
    for m in res.messages:
        if isinstance(m.content, list):
            for b in m.content:
                if isinstance(b, ToolResultBlock) and isinstance(b.content, str):
                    out.append(b.content)
    return out
