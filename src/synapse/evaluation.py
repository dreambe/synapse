"""Eval harness: measure agent behavior so changes are verifiable.

Define cases with checks, run them against an agent, get a pass-rate report.
Checks may be simple (``contains``/``equals``/``matches``), arbitrary
predicates over the :class:`~synapse.runtime.RunResult`, or an
:func:`llm_judge` that scores against a rubric with a model. Checks may be sync
or async.

    from synapse import Agent, Case, evaluate, contains

    report = evaluate(agent, [
        Case("2+2?", check=contains("4")),
        Case("capital of France?", expect_contains="Paris"),
    ])
    print(report.summary())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional, Union

from ._util import maybe_await
from .runtime import RunResult, run_sync

Check = Callable[[RunResult], Union[bool, tuple, Awaitable]]


@dataclass
class Case:
    """One evaluation case."""

    input: Any
    name: str = ""
    check: Optional[Check] = None
    expect_contains: Optional[str] = None
    run_kwargs: dict = field(default_factory=dict)


@dataclass
class CaseResult:
    name: str
    passed: bool
    detail: str
    output: str
    tokens: int
    stop_reason: str = "end_turn"


@dataclass
class Report:
    results: list[CaseResult]

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total else 0.0

    @property
    def total_tokens(self) -> int:
        return sum(r.tokens for r in self.results)

    def __bool__(self) -> bool:
        return self.total > 0 and self.passed == self.total

    def summary(self) -> str:
        lines = [f"{'PASS' if r.passed else 'FAIL'}  {r.name}" + (f"  — {r.detail}" if r.detail else "") for r in self.results]
        lines.append(f"\n{self.passed}/{self.total} passed ({self.pass_rate:.0%}), {self.total_tokens} tokens")
        return "\n".join(lines)


async def _run_check(case: Case, result: RunResult) -> tuple[bool, str]:
    if case.check is not None:
        out = await maybe_await(case.check(result))
        if isinstance(out, tuple):
            return bool(out[0]), str(out[1])
        return bool(out), ""
    if case.expect_contains is not None:
        ok = case.expect_contains in result.output
        return ok, f"contains {case.expect_contains!r}"
    return True, "(no check)"


async def aevaluate(agent, cases: list[Case], *, run_kwargs: dict | None = None) -> Report:
    """Run each case against ``agent`` and collect a :class:`Report`.

    ``run_kwargs`` are default run options applied to every case (a case's own
    ``run_kwargs`` override them) — used by Self-Harness to evaluate a harness's
    run policy.
    """
    results: list[CaseResult] = []
    for case in cases:
        merged = {**(run_kwargs or {}), **case.run_kwargs}
        result = await agent.arun(case.input, **merged)
        ok, detail = await _run_check(case, result)
        name = case.name or (case.input if isinstance(case.input, str) else "case")[:48]
        results.append(
            CaseResult(name, ok, detail, result.output, result.usage.total_tokens, result.stop_reason)
        )
    return Report(results)


def evaluate(agent, cases: list[Case], *, run_kwargs: dict | None = None) -> Report:
    """Synchronous wrapper around :func:`aevaluate`."""
    return run_sync(aevaluate(agent, cases, run_kwargs=run_kwargs))


# -- built-in checks --------------------------------------------------------


def contains(text: str, *, case_sensitive: bool = True) -> Check:
    def check(result: RunResult):
        hay = result.output if case_sensitive else result.output.lower()
        needle = text if case_sensitive else text.lower()
        return needle in hay, f"contains {text!r}"

    return check


def equals(text: str) -> Check:
    def check(result: RunResult):
        return result.output.strip() == text, f"equals {text!r}"

    return check


def matches(pattern: str) -> Check:
    compiled = re.compile(pattern)
    def check(result: RunResult):
        return bool(compiled.search(result.output)), f"matches {pattern!r}"

    return check


def llm_judge(model, rubric: str) -> Check:
    """Grade the output against ``rubric`` using ``model`` (PASS/FAIL)."""
    from .messages import Message

    async def check(result: RunResult):
        prompt = (
            f"Rubric:\n{rubric}\n\nResponse to grade:\n{result.output}\n\n"
            "Reply with PASS or FAIL on the first line, then a short reason."
        )
        resp = await model.generate(
            system="You are a strict grader. Begin your reply with PASS or FAIL.",
            messages=[Message("user", prompt)],
            tools=[],
        )
        text = resp.message.text.strip()
        return text.upper().startswith("PASS"), text[:200]

    return check
