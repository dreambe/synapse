# synapse documentation

> ⚠️ Experimental (v0.3). See the README's *Status & honest limitations*.

synapse is a composable agent framework: build an agent once, call it
anywhere — in-process, streaming, or over the **A2A protocol**.

## Guides

- [Getting started](getting-started.md) — install, your first agent, sync/async/streaming.
- [Guide](guide.md) — tools, multimodal, `RunContext` options, memory, context compaction, hooks, guardrails, approval, verifier, routing, teams, resilience.
- [Agent-to-agent (A2A)](a2a.md) — serve to / consume from the A2A ecosystem, push notifications.
- [Self-Harness](self-harness.md) — a versioned harness that improves itself through a regression gate.
- [Enterprise knowledge & escalation](enterprise.md) — ground tasks in a KG (via MCP) and escalate to humans.

## The 60-second tour

```python
from synapse import Agent, tool

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

agent = Agent("calc", instructions="You do arithmetic.", tools=[add])

# sync
print(agent.run("what is 19 * 23 + 5?").output)

# async
result = await agent.arun("...")
print(result.usage.total_tokens)  # token observability

# streaming
async for ev in agent.astream("..."):
    if ev.type == "text_delta":
        print(ev.text, end="")
```

Everything beyond the basics is an opt-in keyword argument on `run`/`arun`/
`astream`, or a field on the canonical [`RunContext`](guide.md#runcontext).
