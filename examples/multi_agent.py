"""Multi-agent composition: a coordinator delegates to specialists in-process.

Each specialist is exposed to the coordinator via ``as_tool()`` — the
in-process form of agent-to-agent communication.

    python examples/multi_agent.py
"""

from __future__ import annotations

from synapse import Agent, ScriptedModel, tool


@tool
def search_docs(query: str) -> str:
    """Look up documentation for a query."""
    return f"docs about {query!r}: synapse agents are composable."


researcher = Agent(
    "researcher",
    instructions="You find information using the search_docs tool.",
    description="Researches questions against the docs.",
    tools=[search_docs],
    # Offline: pretend to search, then summarize.
    model=ScriptedModel(
        [[("search_docs", {"query": "agents"})], "Agents in synapse are composable units."]
    ),
)

writer = Agent(
    "writer",
    instructions="You turn findings into a crisp one-liner.",
    description="Writes concise summaries.",
    model=ScriptedModel(["In short: synapse agents compose cleanly."]),
)

coordinator = Agent(
    "coordinator",
    instructions=(
        "Delegate research to the researcher, then ask the writer to summarize. "
        "Return the writer's summary."
    ),
    tools=[researcher.as_tool(), writer.as_tool()],
    model=ScriptedModel(
        [
            [("ask_researcher", {"input": "What are synapse agents?"})],
            [("ask_writer", {"input": "Summarize: agents are composable."})],
            "Done — synapse agents compose cleanly.",
        ]
    ),
)


if __name__ == "__main__":
    result = coordinator.run("Explain what synapse agents are, briefly.")
    print(result.output)
