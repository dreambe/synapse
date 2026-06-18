"""Benchmark: a versioned, comparable evaluation *standard*.

``evaluation.py`` answers a per-run question — "did this run pass its checks?".
A benchmark answers a longer-lived one: **"is this iteration better or worse
than the last?"** — across many capabilities, with a persisted baseline so a
regression is caught instead of shipped.

The pieces:

- :class:`Probe` — one dimension-tagged, self-contained check returning
  ``(passed, detail)``. :func:`agent_probes` adapts ordinary evaluation
  :class:`~synapse.evaluation.Case` s into probes for scoring *your* agent.
- :class:`Benchmark` — a **named, versioned** suite of probes. Running it
  yields a :class:`Scorecard`.
- :class:`Scorecard` — per-dimension pass rates + an overall score, tagged with
  the benchmark version, the synapse version, the model, and a timestamp;
  serializable so it can be saved as a baseline.
- :func:`compare` — diffs two scorecards and flags **regressions** per
  dimension and per case. This is the mechanism that makes iteration
  measurable; wire it into CI or :mod:`~synapse.selfharness` to gate promotion.

Honest scope: the bundled :func:`synapse_benchmark` measures framework
**mechanism correctness** (loop guards, idempotency, plan tracking, context
offload, structured output, tenant isolation, …) exercised deterministically
with :class:`~synapse.models.ScriptedModel`. It is the framework's regression
spine — **not** a leaderboard of model intelligence. A model-quality benchmark
is a different suite you build with :func:`agent_probes` over real cases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Awaitable, Callable

from ._util import maybe_await
from .runtime import run_sync

# A probe body returns (passed, detail). May be sync or async.
ProbeBody = Callable[[], "bool | tuple | Awaitable"]


@dataclass
class Probe:
    """One dimension-tagged, self-contained benchmark check."""

    name: str
    dimension: str
    body: ProbeBody
    weight: float = 1.0


@dataclass
class ProbeResult:
    name: str
    dimension: str
    passed: bool
    detail: str
    weight: float = 1.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dimension": self.dimension,
            "passed": self.passed,
            "detail": self.detail,
            "weight": self.weight,
        }


@dataclass
class DimensionScore:
    dimension: str
    passed: float
    total: float

    @property
    def score(self) -> float:
        return self.passed / self.total if self.total else 0.0

    def to_dict(self) -> dict:
        return {"dimension": self.dimension, "passed": self.passed, "total": self.total}


@dataclass
class Scorecard:
    """The result of running a :class:`Benchmark` once — a comparable snapshot."""

    benchmark: str
    benchmark_version: str
    synapse_version: str
    model: str
    timestamp: str
    results: list[ProbeResult] = field(default_factory=list)

    @property
    def dimensions(self) -> list[DimensionScore]:
        agg: dict[str, list[float]] = {}
        for r in self.results:
            passed, total = agg.setdefault(r.dimension, [0.0, 0.0])
            agg[r.dimension] = [passed + (r.weight if r.passed else 0.0), total + r.weight]
        return [DimensionScore(d, p, t) for d, (p, t) in sorted(agg.items())]

    @property
    def overall(self) -> float:
        total = sum(r.weight for r in self.results)
        passed = sum(r.weight for r in self.results if r.passed)
        return passed / total if total else 0.0

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "benchmark_version": self.benchmark_version,
            "synapse_version": self.synapse_version,
            "model": self.model,
            "timestamp": self.timestamp,
            "overall": self.overall,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "results": [r.to_dict() for r in self.results],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Scorecard":
        return cls(
            benchmark=d["benchmark"],
            benchmark_version=d["benchmark_version"],
            synapse_version=d["synapse_version"],
            model=d.get("model", ""),
            timestamp=d.get("timestamp", ""),
            results=[
                ProbeResult(
                    name=r["name"],
                    dimension=r["dimension"],
                    passed=r["passed"],
                    detail=r.get("detail", ""),
                    weight=r.get("weight", 1.0),
                )
                for r in d.get("results", [])
            ],
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Scorecard":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def summary(self) -> str:
        lines = [
            f"{self.benchmark} v{self.benchmark_version} "
            f"(synapse {self.synapse_version}, model={self.model or '?'})",
            "",
        ]
        for d in self.dimensions:
            lines.append(f"  {d.score:>5.0%}  {d.dimension}  ({int(d.passed)}/{int(d.total)})")
        passed = sum(1 for r in self.results if r.passed)
        lines.append("")
        lines.append(f"  overall {self.overall:.0%}  ({passed}/{len(self.results)} probes)")
        return "\n".join(lines)


@dataclass
class Benchmark:
    """A named, versioned suite of probes."""

    name: str
    version: str
    probes: list[Probe]
    description: str = ""

    async def arun(self, *, model: str = "") -> Scorecard:
        from . import __version__

        results: list[ProbeResult] = []
        for probe in self.probes:
            try:
                out = await maybe_await(probe.body())
            except Exception as exc:  # noqa: BLE001 - a crashing probe is a failure, not a stop
                results.append(
                    ProbeResult(probe.name, probe.dimension, False,
                                f"raised {type(exc).__name__}: {exc}", probe.weight)
                )
                continue
            if isinstance(out, tuple):
                passed, detail = bool(out[0]), str(out[1])
            else:
                passed, detail = bool(out), ""
            results.append(ProbeResult(probe.name, probe.dimension, passed, detail, probe.weight))
        return Scorecard(
            benchmark=self.name,
            benchmark_version=self.version,
            synapse_version=__version__,
            model=model,
            timestamp=datetime.now(timezone.utc).isoformat(),
            results=results,
        )

    def run(self, *, model: str = "") -> Scorecard:
        """Synchronous wrapper around :meth:`arun`."""
        return run_sync(self.arun(model=model))


# -- comparing iterations ---------------------------------------------------


@dataclass
class DimensionDelta:
    dimension: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    @property
    def regressed(self) -> bool:
        return self.current < self.baseline


@dataclass
class ScorecardDiff:
    """The difference between a baseline scorecard and a new one."""

    overall_baseline: float
    overall_current: float
    dimensions: list[DimensionDelta]
    newly_failing: list[str]
    newly_passing: list[str]

    @property
    def overall_delta(self) -> float:
        return self.overall_current - self.overall_baseline

    @property
    def regressed(self) -> bool:
        """True if any dimension dropped or any case that passed now fails."""
        return bool(self.newly_failing) or any(d.regressed for d in self.dimensions)

    def summary(self) -> str:
        arrow = {True: "▼", False: "▲"}
        lines = [
            f"overall {self.overall_baseline:.0%} → {self.overall_current:.0%} "
            f"({self.overall_delta:+.0%})",
            "",
        ]
        for d in self.dimensions:
            if abs(d.delta) < 1e-9:
                mark = "="
            else:
                mark = arrow[d.regressed]
            lines.append(
                f"  {mark} {d.dimension}: {d.baseline:.0%} → {d.current:.0%} ({d.delta:+.0%})"
            )
        if self.newly_failing:
            lines.append("")
            lines.append("  REGRESSED: " + ", ".join(self.newly_failing))
        if self.newly_passing:
            lines.append("  FIXED: " + ", ".join(self.newly_passing))
        lines.append("")
        lines.append("  REGRESSION" if self.regressed else "  no regression")
        return "\n".join(lines)


def compare(baseline: Scorecard, current: Scorecard) -> ScorecardDiff:
    """Diff two scorecards: per-dimension deltas + per-case regressions/fixes."""
    base_dims = {d.dimension: d.score for d in baseline.dimensions}
    cur_dims = {d.dimension: d.score for d in current.dimensions}
    dims = [
        DimensionDelta(name, base_dims.get(name, 0.0), cur_dims.get(name, 0.0))
        for name in sorted(set(base_dims) | set(cur_dims))
    ]
    base_pass = {r.name: r.passed for r in baseline.results}
    cur_pass = {r.name: r.passed for r in current.results}
    newly_failing = sorted(n for n, p in cur_pass.items() if base_pass.get(n) and not p)
    newly_passing = sorted(n for n, p in cur_pass.items() if p and base_pass.get(n) is False)
    return ScorecardDiff(
        overall_baseline=baseline.overall,
        overall_current=current.overall,
        dimensions=dims,
        newly_failing=newly_failing,
        newly_passing=newly_passing,
    )


# -- adapting evaluation Cases into probes ----------------------------------


def agent_probes(agent, cases, *, dimension: str = "task", run_kwargs: dict | None = None) -> list[Probe]:
    """Turn evaluation :class:`~synapse.evaluation.Case` s into probes that score
    ``agent``, so you can fold a real task suite into a benchmark and compare it
    across iterations of your agent.
    """
    from .evaluation import _run_check

    def make(case) -> Probe:
        async def body() -> tuple[bool, str]:
            merged = {**(run_kwargs or {}), **case.run_kwargs}
            result = await agent.arun(case.input, **merged)
            return await _run_check(case, result)

        name = case.name or (case.input if isinstance(case.input, str) else "case")[:48]
        dim = getattr(case, "dimension", "") or dimension
        return Probe(name=name, dimension=dim, body=body)

    return [make(c) for c in cases]


# -- the bundled framework regression suite ---------------------------------


def synapse_benchmark() -> Benchmark:
    """The framework's own regression spine.

    Each probe builds a deterministic :class:`~synapse.models.ScriptedModel`
    agent and asserts that *one framework mechanism* behaves as claimed. Fully
    offline. Run it on every iteration and :func:`compare` against a saved
    baseline to catch a capability quietly breaking.
    """
    from .agent import Agent
    from .journal import InMemoryJournal
    from .memory import InMemoryNamespace
    from .messages import DocumentBlock, ToolResultBlock
    from .models import RetryModel, ScriptedModel
    from .results import InMemoryResultStore
    from .tool import tool

    # -- tool_use: a basic tool call round-trips into the answer -------------
    async def p_tool_use() -> tuple[bool, str]:
        @tool
        def add(a: int, b: int) -> int:
            "Add two integers."
            return a + b

        model = ScriptedModel([[("add", {"a": 19, "b": 23})], "The answer is 42."])
        agent = Agent("calc", model=model, tools=[add])
        res = await agent.arun("19+23?")
        return ("42" in res.output, res.output)

    # -- loop_control: a repeating call trips the loop guard -----------------
    async def p_loop_detected() -> tuple[bool, str]:
        @tool
        def noop(x: int) -> str:
            "Does nothing."
            return "ok"

        model = ScriptedModel([[("noop", {"x": 1})]] * 6 + ["done"])
        agent = Agent("looper", model=model, tools=[noop])
        res = await agent.arun("go", max_repeated_tool_calls=2)
        return (res.stop_reason == "loop_detected", res.stop_reason)

    # -- loop_control: repeated tool errors trip the circuit breaker ---------
    async def p_tool_errors() -> tuple[bool, str]:
        @tool
        def boom() -> str:
            "Always fails."
            raise ValueError("nope")

        model = ScriptedModel([[("boom", {})]] * 6 + ["done"])
        agent = Agent("breaker", model=model, tools=[boom])
        res = await agent.arun("go", max_consecutive_tool_errors=2)
        return (res.stop_reason == "tool_errors_exhausted", res.stop_reason)

    # -- idempotency: a journalled side effect fires once across replays -----
    async def p_idempotency() -> tuple[bool, str]:
        calls: list[int] = []

        @tool
        def charge() -> str:
            "A side-effecting action."
            calls.append(1)
            return "charged"

        journal = InMemoryJournal()
        script: list = [[("charge", {})], "done"]
        a1 = Agent("pay", model=ScriptedModel(list(script)), tools=[charge])
        await a1.arun("pay", journal=journal, run_id="r1")
        a2 = Agent("pay", model=ScriptedModel(list(script)), tools=[charge])
        await a2.arun("pay", journal=journal, run_id="r1")
        return (len(calls) == 1, f"side effect fired {len(calls)}x")

    # -- planning: plan=True yields a tracked decomposition ------------------
    async def p_planning() -> tuple[bool, str]:
        model = ScriptedModel(
            [[("write_plan", {"steps": ["research", "write", "review"]})], "done"]
        )
        agent = Agent("planner", model=model)
        res = await agent.arun("do a thing", plan=True)
        n = len(res.plan.steps) if res.plan else 0
        return (n == 3, f"{n} plan steps tracked")

    # -- context_offload: a large result is offloaded with a reference -------
    async def p_offload() -> tuple[bool, str]:
        big = "x" * 5000

        @tool
        def dump() -> str:
            "Returns a large blob."
            return big

        store = InMemoryResultStore()
        model = ScriptedModel([[("dump", {})], "done"])
        agent = Agent("offloader", model=model, tools=[dump])
        res = await agent.arun("dump", offload_over=100, result_store=store)
        offloaded = any(
            isinstance(b, ToolResultBlock)
            and isinstance(b.content, str)
            and "offloaded as res_" in b.content
            for m in res.messages
            if isinstance(m.content, list)
            for b in m.content
        )
        return (offloaded, "large result kept out of context" if offloaded else "not offloaded")

    # -- structured_output: the final answer parses + validates --------------
    async def p_structured() -> tuple[bool, str]:
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}, "pop": {"type": "integer"}},
            "required": ["city", "pop"],
        }
        model = ScriptedModel(['{"city": "Paris", "pop": 2000000}'])
        agent = Agent("structured", model=model)
        res = await agent.arun("the capital?", output_schema=schema)
        ok = isinstance(res.parsed, dict) and res.parsed.get("city") == "Paris"
        return (ok, f"parsed={res.parsed!r}")

    # -- isolation: one tenant never recalls another's memory ----------------
    async def p_isolation() -> tuple[bool, str]:
        ns = InMemoryNamespace()
        await ns.scope("alice").add("alice's secret salary is 100k")
        bob_hits = await ns.scope("bob").search("salary")
        alice_hits = await ns.scope("alice").search("salary")
        ok = not bob_hits and bool(alice_hits)
        return (ok, f"bob saw {len(bob_hits)} of alice's facts")

    # -- resilience: RetryModel fails over to a healthy backend --------------
    async def p_resilience() -> tuple[bool, str]:
        class Flaky(ScriptedModel):
            async def generate(self, *, system, messages, tools):
                raise ConnectionError("primary down")

        primary = Flaky([])
        backup = ScriptedModel(["recovered"])
        agent = Agent("resilient", model=RetryModel(primary, fallbacks=[backup]))
        res = await agent.arun("hi")
        return (res.output == "recovered", res.output)

    # -- multimodal: a tool may return rich content blocks -------------------
    async def p_multimodal() -> tuple[bool, str]:
        @tool
        def fetch_doc() -> DocumentBlock:
            "Returns a document."
            return DocumentBlock.from_text("hello world", title="note.txt")

        model = ScriptedModel([[("fetch_doc", {})], "done"])
        agent = Agent("reader", model=model, tools=[fetch_doc])
        res = await agent.arun("read it")
        rich = any(
            isinstance(b, ToolResultBlock) and isinstance(b.content, list)
            for m in res.messages
            if isinstance(m.content, list)
            for b in m.content
        )
        return (rich, "rich tool result flowed through" if rich else "no rich content")

    probes = [
        Probe("basic_tool_call", "tool_use", p_tool_use),
        Probe("loop_detected", "loop_control", p_loop_detected),
        Probe("tool_errors_exhausted", "loop_control", p_tool_errors),
        Probe("journal_replay", "idempotency", p_idempotency),
        Probe("plan_tracked", "planning", p_planning),
        Probe("large_result_offloaded", "context", p_offload),
        Probe("structured_output_parsed", "structured_output", p_structured),
        Probe("tenant_memory_isolated", "isolation", p_isolation),
        Probe("model_failover", "resilience", p_resilience),
        Probe("rich_tool_result", "multimodal", p_multimodal),
    ]
    return Benchmark(
        name="synapse-core",
        version="1.0",
        probes=probes,
        description="Framework mechanism-correctness regression suite (offline, deterministic).",
    )
