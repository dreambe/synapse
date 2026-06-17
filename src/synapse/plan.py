"""An explicit plan: decomposition as a maintained artifact, not a hope.

A frontier coding agent keeps a visible to-do list — a context anchor and a
progress ledger. ``Plan`` makes that a first-class object the agent edits with
``write_plan`` / ``update_step`` tools; the current plan is rendered into the
agent's context every turn, and survives on ``RunResult.plan``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

PENDING = "pending"
IN_PROGRESS = "in_progress"
DONE = "done"
BLOCKED = "blocked"

_MARK = {PENDING: "[ ]", IN_PROGRESS: "[~]", DONE: "[x]", BLOCKED: "[!]"}


@dataclass
class PlanStep:
    text: str
    status: str = PENDING


@dataclass
class Plan:
    steps: list[PlanStep] = field(default_factory=list)

    def render(self) -> str:
        if not self.steps:
            return "(no plan yet — call write_plan to lay out the steps)"
        return "\n".join(f"{_MARK.get(s.status, '[ ]')} {s.text}" for s in self.steps)

    @property
    def done(self) -> bool:
        return bool(self.steps) and all(s.status == DONE for s in self.steps)

    def to_dict(self) -> dict:
        return {"steps": [{"text": s.text, "status": s.status} for s in self.steps]}


def _find(plan: Plan, step: object) -> PlanStep | None:
    if isinstance(step, int) and 0 <= step < len(plan.steps):
        return plan.steps[step]
    if isinstance(step, str):
        for s in plan.steps:
            if s.text == step:
                return s
        # fall back to a 1-based index passed as a string
        if step.isdigit() and 0 < int(step) <= len(plan.steps):
            return plan.steps[int(step) - 1]
    return None


def plan_tools(plan: Plan):
    """Build the ``write_plan`` / ``update_step`` tools bound to ``plan``."""
    from .tool import Tool

    def write_plan(steps: list) -> str:
        plan.steps = [PlanStep(str(s)) for s in steps]
        return "Plan set:\n" + plan.render()

    def update_step(step: str, status: str) -> str:
        if status not in _MARK:
            return f"invalid status {status!r}; use one of {sorted(_MARK)}"
        target = _find(plan, step)
        if target is None:
            return f"no step matching {step!r}"
        target.status = status
        return "Plan:\n" + plan.render()

    return [
        Tool(
            name="write_plan",
            description="Lay out (or replace) the step-by-step plan for the task.",
            parameters={
                "type": "object",
                "properties": {
                    "steps": {"type": "array", "items": {"type": "string"},
                              "description": "The ordered steps."}
                },
                "required": ["steps"],
            },
            func=write_plan,
        ),
        Tool(
            name="update_step",
            description="Update a plan step's status as you make progress.",
            parameters={
                "type": "object",
                "properties": {
                    "step": {"type": "string", "description": "Step text or 1-based number."},
                    "status": {"type": "string", "enum": [PENDING, IN_PROGRESS, DONE, BLOCKED]},
                },
                "required": ["step", "status"],
            },
            func=update_step,
        ),
    ]
