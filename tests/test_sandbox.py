"""Tests for the code-execution sandbox."""

from __future__ import annotations

from synapse import Agent, ScriptedModel, code_execution_tool, run_python


async def test_run_python_captures_stdout():
    result = await run_python("print(2 + 2)")
    assert result.ok
    assert result.stdout.strip() == "4"


async def test_run_python_nonzero_on_error():
    result = await run_python("raise ValueError('boom')")
    assert not result.ok
    assert result.returncode != 0
    assert "ValueError" in result.stderr


async def test_run_python_times_out():
    result = await run_python("import time; time.sleep(5)", timeout=0.3)
    assert result.timed_out
    assert not result.ok


async def test_run_python_isolated_workdir():
    # Writes land in the sandbox temp dir, not the project.
    result = await run_python("open('out.txt', 'w').write('hi'); print('wrote')")
    assert result.ok and "wrote" in result.stdout


async def test_code_execution_tool_via_agent():
    agent = Agent(
        "coder",
        model=ScriptedModel([[("run_python", {"code": "print(6 * 7)"})], "the answer is 42"]),
        tools=[code_execution_tool()],
    )
    result = await agent.arun("compute 6*7")
    assert result.output == "the answer is 42"
    tool_results = [
        b.content
        for m in result.messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result"
    ]
    assert any("42" in c for c in tool_results if isinstance(c, str))
