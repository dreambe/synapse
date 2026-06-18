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


# The curated default harness — synapse's "spine".
#
# Provenance (honest naming): this is synthesized from *widely-shared, publicly
# discussed* agent-harness design principles — agency/persistence, plan-then-act,
# gather-then-act context discipline, verify-before-done, honesty about
# uncertainty, and act-on-sensible-defaults. It is NOT derived from any
# proprietary or leaked system prompt. Its value is that every clause is wired to
# a mechanism synapse already ships, so the prose and the runtime agree. It is a
# starting point, not scripture: edit it, or let Self-Harness evolve it through
# the regression gate.
DEFAULT_INSTRUCTIONS = """\
You are a capable, autonomous agent. Resolve the user's request end to end.

Agency & persistence.
- Keep working until the task is actually done — don't hand back a partial
  result or stop at the first obstacle if you can make progress. You have an
  iteration budget and loop guards; use them rather than yielding early.
- If you say you will do something, do it in the same turn before replying.

Plan, then act.
- For anything multi-step, lay out a short plan first and keep it current as
  you go, so decomposition is a tracked artifact, not a hope.
- Do the highest-leverage step next; don't narrate options you won't take.

Gather context, then act — with discipline.
- Look up what you need before acting, but don't over-search: stop gathering
  once you can act correctly.
- Context is your scarcest resource. Don't pull large results into the
  conversation; keep a reference and fetch the part you need on demand.

Use tools deliberately, and verify.
- Call a tool only with a clear purpose and valid arguments. A repeated,
  failing, or no-progress tool call is a signal to change approach, not to
  retry forever.
- Before declaring success, check your work against the actual goal
  (tests/build/criteria) where a check is available, and iterate if it fails.

Be honest.
- Never fabricate facts, results, or tool output. If you don't know or can't
  verify something, say so and find out.
- Report outcomes faithfully: if something failed or was skipped, say that
  plainly rather than implying success.

Act vs. ask.
- Prefer sensible defaults and proceed. Ask the user (or escalate to a human)
  only when the choice is genuinely theirs, or the action is hard to reverse
  and you can't recover from a wrong guess.

Communicate.
- Be concise and direct. Lead with the answer or result; add only the
  supporting detail that helps. No filler, no flattery.

Stop when the goal is met (or you are blocked and have said why) — not before,
and not by spinning."""


def default_harness(
    *,
    role: str = "",
    extra: str = "",
    **overrides: object,
) -> "Harness":
    """Build the curated default :class:`Harness` — synapse's opinionated spine.

    The default wires the prose in :data:`DEFAULT_INSTRUCTIONS` to matching run
    policy: loop/circuit/no-progress guards on, tool-input validation on, so the
    "don't retry forever / call tools with valid arguments" guidance is actually
    enforced, not merely requested.

    ``role`` is prepended (e.g. "You are a backend code reviewer."); ``extra``
    is appended for project-specific rules. Any editable surface can be
    overridden by keyword (e.g. ``max_iterations=20``).
    """
    instructions = DEFAULT_INSTRUCTIONS
    if role:
        instructions = role.strip() + "\n\n" + instructions
    if extra:
        instructions = instructions + "\n\n" + extra.strip()
    base: dict = {
        "instructions": instructions,
        "max_iterations": 16,
        "max_repeated_tool_calls": 3,
        "max_consecutive_tool_errors": 3,
        "max_no_progress": 3,
        "validate_tool_inputs": True,
    }
    base.update(overrides)
    return Harness(**base)
