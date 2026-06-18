"""Tests for the benchmark / evaluation-standard layer."""

from __future__ import annotations


from synapse import (
    Agent,
    Benchmark,
    Case,
    Probe,
    Scorecard,
    ScriptedModel,
    agent_probes,
    compare,
    contains,
    synapse_benchmark,
)


def test_synapse_benchmark_all_green():
    """The bundled framework regression suite should pass on a healthy build."""
    card = synapse_benchmark().run(model="ScriptedModel")
    assert card.benchmark == "synapse-core"
    assert card.overall == 1.0, card.summary()
    # every declared dimension is exercised
    dims = {d.dimension for d in card.dimensions}
    assert {"loop_control", "idempotency", "isolation", "context"} <= dims


def test_scorecard_roundtrip_and_dimensions():
    card = synapse_benchmark().run()
    restored = Scorecard.from_dict(card.to_dict())
    assert restored.overall == card.overall
    assert len(restored.results) == len(card.results)
    # dimension aggregation is stable
    loop = next(d for d in card.dimensions if d.dimension == "loop_control")
    assert loop.total == 2  # two loop_control probes


def test_compare_detects_regression():
    baseline = Scorecard("b", "1.0", "0.3.0", "m", "t")
    baseline.results = [
        _r("keep", "core", True),
        _r("breaks", "core", True),
        _r("fixed", "extra", False),
    ]
    current = Scorecard("b", "1.0", "0.3.0", "m", "t")
    current.results = [
        _r("keep", "core", True),
        _r("breaks", "core", False),
        _r("fixed", "extra", True),
    ]
    diff = compare(baseline, current)
    assert diff.regressed is True
    assert diff.newly_failing == ["breaks"]
    assert diff.newly_passing == ["fixed"]
    assert "REGRESSION" in diff.summary()


def test_compare_no_regression_when_improving():
    baseline = Scorecard("b", "1.0", "0.3.0", "m", "t")
    baseline.results = [_r("a", "core", True), _r("b", "core", False)]
    current = Scorecard("b", "1.0", "0.3.0", "m", "t")
    current.results = [_r("a", "core", True), _r("b", "core", True)]
    diff = compare(baseline, current)
    assert diff.regressed is False
    assert diff.overall_delta > 0


def test_agent_probes_adapt_cases():
    agent = Agent("a", model=ScriptedModel(["the answer is 42"] * 3))
    probes = agent_probes(
        agent, [Case("q", check=contains("42"), name="forty-two")], dimension="math"
    )
    assert all(isinstance(p, Probe) for p in probes)
    card = Benchmark("adhoc", "0.1", probes).run()
    assert card.overall == 1.0
    assert card.dimensions[0].dimension == "math"


def test_crashing_probe_is_a_failure_not_a_stop():
    def boom():
        raise RuntimeError("kaboom")

    card = Benchmark("x", "1", [Probe("crash", "core", boom)]).run()
    assert card.overall == 0.0
    assert "kaboom" in card.results[0].detail


def test_scorecard_save_load(tmp_path):
    card = synapse_benchmark().run()
    p = tmp_path / "baseline.json"
    card.save(p)
    loaded = Scorecard.load(p)
    assert loaded.overall == card.overall


def _r(name, dim, passed):
    from synapse import ProbeResult

    return ProbeResult(name, dim, passed, "")
