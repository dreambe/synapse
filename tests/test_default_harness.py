"""Tests for the curated default harness — synapse's opinionated spine."""

from __future__ import annotations

from synapse import (
    DEFAULT_INSTRUCTIONS,
    Agent,
    ScriptedModel,
    default_harness,
)


def test_default_harness_wires_prose_to_policy():
    h = default_harness()
    # The run policy enforces the prose: guards on, validation on.
    assert h.max_repeated_tool_calls == 3
    assert h.max_consecutive_tool_errors == 3
    assert h.max_no_progress == 3
    assert h.validate_tool_inputs is True
    # run_kwargs() is what the agent actually runs with.
    kw = h.run_kwargs()
    assert kw["validate_tool_inputs"] is True
    assert kw["max_repeated_tool_calls"] == 3


def test_default_harness_instructions_are_the_default():
    assert default_harness().instructions == DEFAULT_INSTRUCTIONS
    assert "Agency & persistence." in DEFAULT_INSTRUCTIONS
    assert "Be honest." in DEFAULT_INSTRUCTIONS


def test_role_and_extra_compose():
    h = default_harness(role="You are a code reviewer.", extra="Never touch prod.")
    assert h.instructions.startswith("You are a code reviewer.")
    assert h.instructions.endswith("Never touch prod.")
    assert DEFAULT_INSTRUCTIONS in h.instructions


def test_overrides_apply_and_stay_editable():
    h = default_harness(max_iterations=20)
    assert h.max_iterations == 20
    # still a normal Harness: editable + version-bumping
    h2 = h.with_edits({"max_iterations": 5})
    assert h2.max_iterations == 5 and h2.version == h.version + 1


def test_builds_a_runnable_agent():
    h = default_harness(role="You answer tersely.")
    agent = h.agent("assistant", ScriptedModel(["ok"]))
    assert isinstance(agent, Agent)
    assert agent.instructions.startswith("You answer tersely.")
    assert agent.run("hi", **h.run_kwargs()).output == "ok"
