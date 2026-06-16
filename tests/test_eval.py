"""Tests for the eval harness."""

from __future__ import annotations

from synapse import (
    Agent,
    Case,
    ScriptedModel,
    contains,
    equals,
    evaluate,
    llm_judge,
    matches,
)


def test_contains_equals_matches_and_expect():
    agent = Agent("a", model=ScriptedModel(["the answer is 42"] * 10))
    report = evaluate(
        agent,
        [
            Case("q", check=contains("42")),
            Case("q", check=equals("the answer is 42")),
            Case("q", check=matches(r"\d+")),
            Case("q", expect_contains="answer"),
        ],
    )
    assert report.total == 4 and report.passed == 4
    assert report.pass_rate == 1.0
    assert bool(report) is True


def test_failures_counted():
    agent = Agent("a", model=ScriptedModel(["nope"] * 3))
    report = evaluate(agent, [Case("q", check=contains("yes")), Case("q", expect_contains="nope")])
    assert report.passed == 1 and report.total == 2
    assert not report
    assert "FAIL" in report.summary() and "PASS" in report.summary()


def test_predicate_check_with_detail():
    agent = Agent("a", model=ScriptedModel(["short"]))
    report = evaluate(agent, [Case("q", check=lambda r: (len(r.output) < 10, "len ok"))])
    assert report.passed == 1
    assert report.results[0].detail == "len ok"


def test_llm_judge():
    # The agent under test, and a separate judge model.
    agent = Agent("a", model=ScriptedModel(["Paris is the capital of France."]))
    judge_model = ScriptedModel(["PASS — correctly identifies Paris"])
    report = evaluate(agent, [Case("capital?", check=llm_judge(judge_model, "Must name Paris."))])
    assert report.passed == 1
    assert "Paris" in report.results[0].detail


def test_llm_judge_fail():
    agent = Agent("a", model=ScriptedModel(["London"]))
    judge_model = ScriptedModel(["FAIL — wrong city"])
    report = evaluate(agent, [Case("capital of France?", check=llm_judge(judge_model, "Must say Paris."))])
    assert report.passed == 0
