"""Tests for structured outputs and tool-input validation."""

from __future__ import annotations

from synapse import Agent, ScriptedModel, parse_output, tool, validate_json


def test_validate_types_required_enum():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "role": {"enum": ["a", "b"]},
        },
        "required": ["name", "age"],
    }
    assert validate_json({"name": "x", "age": 3, "role": "a"}, schema) == []
    assert validate_json({"name": "x"}, schema)  # missing age
    assert validate_json({"name": "x", "age": "3"}, schema)  # wrong type
    assert validate_json({"name": "x", "age": 3, "role": "z"}, schema)  # bad enum


def test_bool_is_not_integer():
    assert validate_json(True, {"type": "integer"})  # bool rejected as integer


def test_parse_output_extracts_fenced_json():
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    parsed, errs = parse_output('```json\n{"x": 5}\n```', schema)
    assert parsed == {"x": 5} and errs == []


def test_parse_output_invalid_returns_errors():
    schema = {"type": "object", "required": ["x"]}
    parsed, errs = parse_output("{}", schema)
    assert parsed is None and errs


async def test_agent_output_schema_populates_parsed():
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    agent = Agent("x", model=ScriptedModel(['{"name": "Alice", "age": 30}']))
    result = await agent.arun("profile?", output_schema=schema)
    assert result.parsed == {"name": "Alice", "age": 30}


async def test_agent_output_schema_invalid_parsed_none():
    schema = {"type": "object", "required": ["name"]}
    agent = Agent("x", model=ScriptedModel(["not json at all"]))
    result = await agent.arun("profile?", output_schema=schema)
    assert result.parsed is None


@tool
def add(a: int, b: int) -> int:
    """Add."""
    return a + b


async def test_tool_input_validation_rejects_bad_args():
    agent = Agent(
        "x",
        model=ScriptedModel([[("add", {"a": "oops", "b": 2})], "recovered"]),
        tools=[add],
    )
    result = await agent.arun("add", validate_tool_inputs=True)
    errors = [
        b
        for m in result.messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result" and b.is_error
    ]
    assert errors and "Invalid tool input" in errors[0].content


async def test_tool_input_validation_allows_good_args():
    calls = []

    @tool
    def rec(a: int) -> str:
        """Record."""
        calls.append(a)
        return "ok"

    agent = Agent("x", model=ScriptedModel([[("rec", {"a": 5})], "done"]), tools=[rec])
    await agent.arun("go", validate_tool_inputs=True)
    assert calls == [5]
