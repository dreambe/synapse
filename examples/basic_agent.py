"""A minimal agent with two tools.

Run it (needs an Anthropic API key):

    export ANTHROPIC_API_KEY=...
    synapse run examples/basic_agent.py:agent "What is 19 * 23, then minus 7?"

Or offline, with the scripted backend:

    python examples/basic_agent.py
"""

from __future__ import annotations

from synapse import Agent, tool


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


agent = Agent(
    "calculator",
    instructions="You are a precise calculator. Use the tools for all arithmetic.",
    description="Does exact integer arithmetic using tools.",
    tools=[add, multiply],
)


if __name__ == "__main__":
    # Offline demo using the scripted backend — no API key required.
    from synapse import ScriptedModel

    agent.model = ScriptedModel(
        [
            [("multiply", {"a": 19, "b": 23})],
            [("add", {"a": 437, "b": -7})],
            "19 * 23 - 7 = 430",
        ]
    )
    print(agent.run("What is 19 * 23, then minus 7?").output)
