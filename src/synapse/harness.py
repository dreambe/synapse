"""The Harness: the agent's editable, versioned runtime surface.

A *harness* bundles everything around the model that an agent's behavior
depends on — instructions, tool descriptions, and the run policy (iteration cap,
loop guards, input validation) — into one versioned object. Self-Harness
(:mod:`synapse.selfharness`) proposes edits to *declared* surfaces only and
promotes them through a regression gate, so the harness becomes a managed,
versioned object rather than an ever-thickening prompt.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Optional

from .tool import Tool

if TYPE_CHECKING:
    from .agent import Agent
    from .models import Model

# The only surfaces a proposal may touch. Permissions, billing, external
# connector auth, etc. stay out of bounds by construction.
EDITABLE_SURFACES = frozenset(
    {
        "instructions",
        "tool_descriptions",
        "max_iterations",
        "max_repeated_tool_calls",
        "max_consecutive_tool_errors",
        "max_no_progress",
        "validate_tool_inputs",
    }
)


@dataclass
class Harness:
    """A versioned, declared runtime surface for an agent."""

    instructions: str = ""
    tool_descriptions: dict[str, str] = field(default_factory=dict)
    max_iterations: int = 12
    max_repeated_tool_calls: Optional[int] = None
    max_consecutive_tool_errors: Optional[int] = None
    max_no_progress: Optional[int] = None
    validate_tool_inputs: bool = False
    version: int = 1

    def surfaces(self) -> dict:
        """The current value of every editable surface."""
        return {k: getattr(self, k) for k in EDITABLE_SURFACES}

    def run_kwargs(self) -> dict:
        """Run options this harness imposes (the run-policy surfaces)."""
        return {
            "max_iterations": self.max_iterations,
            "max_repeated_tool_calls": self.max_repeated_tool_calls,
            "max_consecutive_tool_errors": self.max_consecutive_tool_errors,
            "max_no_progress": self.max_no_progress,
            "validate_tool_inputs": self.validate_tool_inputs,
        }

    def agent(self, name: str, model: "Model", tools: tuple[Tool, ...] = ()) -> "Agent":
        """Build an agent with this harness's instructions and tool descriptions."""
        from .agent import Agent

        applied = [
            replace(t, description=self.tool_descriptions[t.name])
            if t.name in self.tool_descriptions
            else t
            for t in tools
        ]
        return Agent(name, instructions=self.instructions, model=model, tools=applied)

    def with_edits(self, changes: dict) -> "Harness":
        """Return a new, version-bumped harness with ``changes`` applied.

        Raises ``ValueError`` if a change targets a non-editable surface.
        ``tool_descriptions`` is merged; other surfaces are replaced.
        """
        bad = set(changes) - EDITABLE_SURFACES
        if bad:
            raise ValueError(f"non-editable surface(s): {sorted(bad)}")
        data = self.surfaces()
        for key, value in changes.items():
            if key == "tool_descriptions":
                data["tool_descriptions"] = {**data["tool_descriptions"], **value}
            else:
                data[key] = value
        return Harness(version=self.version + 1, **data)
