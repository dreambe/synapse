"""Self-Harness: improve the harness from evidence, gated by regression tests.

Offline demo — a model whose output reflects the harness instructions, so a
proposed edit changes behavior deterministically and the promotion gate is
observable without a real model.

    python examples/self_harness.py
"""

from __future__ import annotations

import asyncio

from synapse import Case, Harness, HarnessEdit, contains, evolve
from synapse.messages import Message, TextBlock
from synapse.models.base import Model, ModelResponse


class ReflectingModel(Model):
    async def generate(self, *, system, messages, tools):
        good = "precise" in (system or "")
        return ModelResponse(Message("assistant", [TextBlock(("GOOD" if good else "BAD") + " answer")]))


def fix_instructions(evidence):
    # A trivial proposer: when answers fail, propose a precise-instructions edit.
    return [
        HarnessEdit(
            target_failure="check_failed",
            edited_surface="instructions",
            expected_effect="produce GOOD answers",
            regression_risk="low",
            changes={"instructions": "Be precise."},
        )
    ]


async def main() -> None:
    result = await evolve(
        harness=Harness(instructions="Answer the question."),
        model=ReflectingModel(),
        held_in=[Case("q1", check=contains("GOOD"))],
        held_out=[Case("q2", check=contains("answer"))],  # passes either way
        proposer=fix_instructions,
        rounds=3,
    )
    print(f"final harness v{result.harness.version}, promotions: {result.promotions}")
    print(f"instructions now: {result.harness.instructions!r}")
    for rec in result.records:
        print(("PROMOTED " if rec.promoted else "rejected ") + rec.edit.target_failure + " — " + rec.reason)


if __name__ == "__main__":
    asyncio.run(main())
