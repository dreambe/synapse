"""Run monitoring + a versioned evaluation standard, side by side.

Two operability primitives:

1. **Monitor** — attach it (``hooks=monitor``) and forward it into sub-agents
   (``as_tool(hooks=monitor)``) to see a *live tree* of who spawned whom and
   what each agent is doing (delegating, calling tools, …). ``monitor_app``
   serves a minimal web dashboard over the same data.

2. **Benchmark** — ``synapse_benchmark()`` scores framework mechanism
   correctness into a comparable ``Scorecard``; ``compare`` diffs two
   scorecards so a regression between iterations is caught, not shipped.

    python examples/observability_and_eval.py
    # dashboard:  uvicorn examples.observability_and_eval:app   (pip install synapse[server])
"""

from __future__ import annotations

import asyncio

from synapse import (
    Agent,
    Monitor,
    ScriptedModel,
    compare,
    monitor_app,
    synapse_benchmark,
    tool,
)


@tool
def search_docs(query: str) -> str:
    """Look up documentation for a query."""
    return f"docs about {query!r}."


def build_team(monitor: Monitor) -> Agent:
    researcher = Agent(
        "researcher",
        description="Researches questions.",
        tools=[search_docs],
        model=ScriptedModel(
            [[("search_docs", {"query": "agents"})], "Agents compose cleanly."]
        ),
    )
    return Agent(
        "coordinator",
        instructions="Delegate research, then summarize.",
        # Forwarding the monitor lets us see *inside* the sub-agent.
        tools=[researcher.as_tool(hooks=monitor)],
        model=ScriptedModel(
            [[("ask_researcher", {"input": "what are agents?"})], "Done: agents compose."]
        ),
    )


# A live monitor the dashboard can serve.
monitor = Monitor()
app = monitor_app(monitor)


async def main() -> None:
    # 1) Run a small team under the monitor, then print the agent tree.
    coordinator = build_team(monitor)
    await coordinator.arun("research agents", hooks=monitor)

    print("=== run tree ===")
    for root in monitor.tree():
        print(f"{root['agent']}  ({root['status']}, {root['iterations']} iter)")
        for child in root["children"]:
            print(f"  └─ {child['agent']}  ({child['status']})")
        for ev in root["activity"]:
            if ev["kind"] == "tool_start":
                print(f"     · {ev['agent']} → {ev['detail']}:{ev['name']}")

    # 2) Score the framework, then compare against a (here, identical) baseline.
    print("\n=== benchmark ===")
    card = synapse_benchmark().run(model="ScriptedModel")
    print(card.summary())

    baseline = card  # in practice: Scorecard.load("baseline.json")
    diff = compare(baseline, card)
    print("\n=== vs baseline ===")
    print(diff.summary())


if __name__ == "__main__":
    asyncio.run(main())
