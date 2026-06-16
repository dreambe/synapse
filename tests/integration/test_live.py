"""Live integration tests — proof the framework works against a real model.

Deselected by default (``addopts = -m 'not integration'``). Run explicitly:

    pytest -m integration

Each test self-skips unless the relevant API key is set, so this file is safe
to collect anywhere.
"""

from __future__ import annotations

import os

import pytest

from synapse import Agent, tool

pytestmark = pytest.mark.integration


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_tool_use_live():
    from synapse import AnthropicModel

    agent = Agent(
        "calc",
        instructions="You are a calculator. Always use the add tool for arithmetic.",
        model=AnthropicModel("claude-opus-4-8", max_tokens=1024),
        tools=[add],
    )
    result = await agent.arun("What is 19 + 23? Use the tool, then state the number.")
    assert "42" in result.output
    assert result.usage.total_tokens > 0


@pytest.mark.skipif(not os.getenv("ANTHROPIC_API_KEY"), reason="ANTHROPIC_API_KEY not set")
async def test_anthropic_structured_output_live():
    from synapse import AnthropicModel

    schema = {
        "type": "object",
        "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
        "required": ["city", "country"],
    }
    agent = Agent("geo", model=AnthropicModel("claude-opus-4-8", max_tokens=1024))
    result = await agent.arun("The capital of France.", output_schema=schema)
    assert result.parsed is not None
    assert "paris" in result.parsed["city"].lower()


@pytest.mark.skipif(not os.getenv("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set")
async def test_openai_tool_use_live():
    from synapse import OpenAIModel

    agent = Agent(
        "calc",
        instructions="You are a calculator. Always use the add tool for arithmetic.",
        model=OpenAIModel(os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
        tools=[add],
    )
    result = await agent.arun("What is 19 + 23? Use the tool, then state the number.")
    assert "42" in result.output
