"""Self-Harness: improve the harness from execution evidence, gated by tests.

The mechanism, not magic (model / tools / budget / evaluator stay fixed — only
the harness changes):

1. **Weakness mining** — run the eval suite; for failures, collect structured
   evidence (the failure label / stop_reason, output, detail), not raw logs.
2. **Proposal** — a proposer suggests *bounded* edits to declared harness
   surfaces, each with an audit record (target failure, surface, expected
   effect, regression risk).
3. **Promotion** — re-evaluate the candidate harness on held-in + held-out
   splits. A candidate is promoted only if at least one split improves and
   **neither regresses**; rejected candidates are recorded, the active harness
   is unchanged.

Experimental, and bounded like the paper: a fixed evaluator with a pass-rate
non-regression gate, over a finite editable surface. Higher-risk edits warrant
a stronger acceptance gate than this.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Awaitable, Callable, Union

from ._util import maybe_await
from .evaluation import Case, CaseResult, Report, aevaluate
from .harness import EDITABLE_SURFACES, Harness


@dataclass
class HarnessEdit:
    """A proposed, audited change to declared harness surfaces."""

    target_failure: str
    edited_surface: str
    expected_effect: str
    regression_risk: str
    changes: dict

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "HarnessEdit":
        return cls(
            target_failure=data.get("target_failure", ""),
            edited_surface=data.get("edited_surface", ""),
            expected_effect=data.get("expected_effect", ""),
            regression_risk=data.get("regression_risk", ""),
            changes=data.get("changes", {}),
        )


@dataclass
class EvidencePack:
    """The structured input a proposer sees — not a pile of logs."""

    harness_surfaces: dict
    editable_surfaces: list[str]
    failures: list[dict]
    success_count: int
    rejected_history: list[dict] = field(default_factory=list)


@dataclass
class PromotionRecord:
    edit: HarnessEdit
    promoted: bool
    reason: str
    held_in_before: float
    held_in_after: float
    held_out_before: float
    held_out_after: float


@dataclass
class EvolveResult:
    harness: Harness
    records: list[PromotionRecord]

    @property
    def promotions(self) -> int:
        return sum(1 for r in self.records if r.promoted)


Proposer = Callable[[EvidencePack], Union[list[HarnessEdit], Awaitable]]


def classify_failure(result: CaseResult) -> list[str]:
    """A small, action-oriented failure taxonomy for one failing case."""
    labels: list[str] = []
    if result.stop_reason not in ("end_turn", "verified"):
        labels.append(result.stop_reason)  # e.g. loop_detected, no_progress, max_iterations
    if not result.passed:
        labels.append("check_failed")
    if not (result.output or "").strip():
        labels.append("no_output")
    return labels or ["unknown"]


def mine_weaknesses(report: Report) -> list[dict]:
    """Structured evidence for each failing case in a report."""
    return [
        {
            "case": r.name,
            "stop_reason": r.stop_reason,
            "labels": classify_failure(r),
            "detail": r.detail,
            "output": (r.output or "")[:240],
        }
        for r in report.results
        if not r.passed
    ]


def _promotes(bi: float, bo: float, ci: float, co: float) -> tuple[bool, str]:
    span = f"held-in {bi:.0%}->{ci:.0%}, held-out {bo:.0%}->{co:.0%}"
    if ci < bi or co < bo:
        return False, f"regression rejected ({span})"
    if ci > bi or co > bo:
        return True, f"promoted ({span})"
    return False, f"no improvement ({span})"


async def evolve(
    *,
    harness: Harness,
    model,
    tools: tuple = (),
    held_in: list[Case],
    held_out: list[Case],
    proposer: Proposer,
    rounds: int = 3,
    name: str = "agent",
) -> EvolveResult:
    """Run the propose→validate→promote loop for up to ``rounds`` rounds."""
    active = harness
    records: list[PromotionRecord] = []
    rejected: list[dict] = []

    for _ in range(rounds):
        base_in = await aevaluate(active.agent(name, model, tools), held_in, run_kwargs=active.run_kwargs())
        base_out = await aevaluate(active.agent(name, model, tools), held_out, run_kwargs=active.run_kwargs())

        evidence = EvidencePack(
            harness_surfaces=active.surfaces(),
            editable_surfaces=sorted(EDITABLE_SURFACES),
            failures=mine_weaknesses(base_in) + mine_weaknesses(base_out),
            success_count=base_in.passed + base_out.passed,
            rejected_history=list(rejected),
        )
        edits = await maybe_await(proposer(evidence))

        progressed = False
        for edit in edits:
            try:
                candidate = active.with_edits(edit.changes)
            except ValueError:
                rejected.append({"target": edit.target_failure, "reason": "non-editable surface"})
                continue
            ci = (
                await aevaluate(candidate.agent(name, model, tools), held_in, run_kwargs=candidate.run_kwargs())
            ).pass_rate
            co = (
                await aevaluate(candidate.agent(name, model, tools), held_out, run_kwargs=candidate.run_kwargs())
            ).pass_rate
            ok, reason = _promotes(base_in.pass_rate, base_out.pass_rate, ci, co)
            records.append(
                PromotionRecord(edit, ok, reason, base_in.pass_rate, ci, base_out.pass_rate, co)
            )
            if ok:
                active = candidate
                progressed = True
                break
            rejected.append({"target": edit.target_failure, "reason": reason})

        if not progressed:
            break  # nothing promoted this round → stop

    return EvolveResult(active, records)


def model_proposer(model, *, max_edits: int = 2) -> Proposer:
    """A proposer that asks a model for bounded harness edits (structured JSON)."""
    from .agent import Agent

    schema = {
        "type": "object",
        "properties": {
            "edits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "target_failure": {"type": "string"},
                        "edited_surface": {"type": "string"},
                        "expected_effect": {"type": "string"},
                        "regression_risk": {"type": "string"},
                        "changes": {"type": "object"},
                    },
                    "required": ["target_failure", "edited_surface", "changes"],
                },
            }
        },
        "required": ["edits"],
    }

    async def propose(evidence: EvidencePack) -> list[HarnessEdit]:
        import json

        prompt = (
            "You improve an agent harness. Propose at most "
            f"{max_edits} minimal, safe edits.\n\n"
            f"Editable surfaces (only these): {evidence.editable_surfaces}\n"
            f"Current surfaces: {json.dumps(evidence.harness_surfaces, default=str)}\n"
            f"Failures: {json.dumps(evidence.failures, default=str)}\n"
            f"Previously rejected: {json.dumps(evidence.rejected_history, default=str)}\n\n"
            "Each edit targets one failure mechanism and changes one surface. "
            "'changes' maps surface names to new values."
        )
        agent = Agent(
            "harness-proposer",
            instructions="You are a careful change manager for agent harnesses.",
            model=model,
        )
        result = await agent.arun(prompt, output_schema=schema)
        data = result.parsed or {}
        raw = data.get("edits", []) if isinstance(data, dict) else []
        return [HarnessEdit.from_dict(e) for e in raw[:max_edits]]

    return propose
