"""Tests for the Self-Harness mechanism (versioned harness + regression gate)."""

from __future__ import annotations

import pytest

from synapse import (
    Case,
    Harness,
    HarnessEdit,
    ScriptedModel,
    contains,
    evolve,
    mine_weaknesses,
    model_proposer,
)
from synapse.evaluation import aevaluate
from synapse.messages import Message, TextBlock
from synapse.models.base import Model, ModelResponse


class SurfaceModel(Model):
    """Output reflects the harness `instructions` — so an edit changes behavior
    deterministically, letting us test the promotion gate offline."""

    async def generate(self, *, system, messages, tools):
        good = "precise" in (system or "")
        return ModelResponse(Message("assistant", [TextBlock(("GOOD" if good else "BAD") + " A")]))


# -- Harness object ----------------------------------------------------------


def test_with_edits_bumps_version_and_merges():
    h = Harness(instructions="x", tool_descriptions={"a": "A"})
    h2 = h.with_edits({"instructions": "y", "tool_descriptions": {"b": "B"}})
    assert h2.version == 2
    assert h2.instructions == "y"
    assert h2.tool_descriptions == {"a": "A", "b": "B"}
    assert h.instructions == "x"  # original unchanged


def test_with_edits_rejects_non_editable_surface():
    with pytest.raises(ValueError):
        Harness().with_edits({"api_key": "secret"})


def test_run_kwargs_carries_loop_policy():
    h = Harness(max_repeated_tool_calls=2, max_iterations=7)
    kw = h.run_kwargs()
    assert kw["max_repeated_tool_calls"] == 2 and kw["max_iterations"] == 7


# -- weakness mining ---------------------------------------------------------


async def test_mine_weaknesses_labels_failures():
    agent = Harness(instructions="").agent("a", SurfaceModel())
    report = await aevaluate(agent, [Case("q", check=contains("GOOD"))])
    weaknesses = mine_weaknesses(report)
    assert len(weaknesses) == 1
    assert "check_failed" in weaknesses[0]["labels"]


# -- the closed loop ---------------------------------------------------------


def _fix_instructions_proposer(value: str):
    async def propose(evidence):
        return [
            HarnessEdit(
                target_failure="check_failed",
                edited_surface="instructions",
                expected_effect="produce GOOD output",
                regression_risk="low",
                changes={"instructions": value},
            )
        ]

    return propose


async def test_evolve_promotes_improving_edit():
    held_in = [Case("q", check=contains("GOOD"))]   # fails until instructions fixed
    held_out = [Case("q", check=contains("A"))]      # always passes (no regression)
    result = await evolve(
        harness=Harness(instructions=""),
        model=SurfaceModel(),
        held_in=held_in,
        held_out=held_out,
        proposer=_fix_instructions_proposer("be precise"),
        rounds=2,
    )
    assert result.promotions == 1
    assert "precise" in result.harness.instructions
    assert result.harness.version == 2
    assert result.records[0].promoted and "promoted" in result.records[0].reason


async def test_evolve_rejects_regressing_edit():
    # Start from a good harness; the proposed edit removes "precise" → regresses.
    held_in = [Case("q", check=contains("GOOD"))]
    held_out = [Case("q", check=contains("GOOD"))]
    result = await evolve(
        harness=Harness(instructions="be precise"),
        model=SurfaceModel(),
        held_in=held_in,
        held_out=held_out,
        proposer=_fix_instructions_proposer("be vague"),
        rounds=2,
    )
    assert result.promotions == 0
    assert result.harness.instructions == "be precise"  # active unchanged
    assert not result.records[0].promoted
    assert "regression" in result.records[0].reason


async def test_evolve_skips_non_editable_edits():
    async def bad_proposer(evidence):
        return [
            HarnessEdit("x", "api_key", "", "", {"api_key": "secret"})
        ]

    result = await evolve(
        harness=Harness(instructions=""),
        model=SurfaceModel(),
        held_in=[Case("q", check=contains("GOOD"))],
        held_out=[Case("q", check=contains("A"))],
        proposer=bad_proposer,
        rounds=1,
    )
    assert result.promotions == 0
    assert result.harness.version == 1  # never changed


async def test_model_proposer_parses_structured_edits():
    # The proposer model returns a JSON edit (dogfoods structured outputs).
    judge_json = (
        '{"edits": [{"target_failure": "check_failed", "edited_surface": "instructions", '
        '"expected_effect": "GOOD", "regression_risk": "low", '
        '"changes": {"instructions": "be precise"}}]}'
    )
    proposer = model_proposer(ScriptedModel([judge_json]))
    from synapse import EvidencePack

    edits = await proposer(
        EvidencePack(harness_surfaces={}, editable_surfaces=["instructions"], failures=[], success_count=0)
    )
    assert len(edits) == 1
    assert edits[0].changes == {"instructions": "be precise"}
