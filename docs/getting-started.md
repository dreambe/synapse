# Getting started

## Install

```bash
pip install synapse                 # core (stdlib only)
pip install "synapse[anthropic]"    # + Claude backend
pip install "synapse[server]"       # + uvicorn (async high-concurrency serving)
pip install "synapse[mcp]"          # + Model Context Protocol client
```

Python 3.10+.

## Your first agent

```python
from synapse import Agent, tool

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

agent = Agent(
    "calculator",
    instructions="You are a precise calculator. Use the tools for arithmetic.",
    tools=[add],
)

print(agent.run("What is 19 + 23?").output)
```

By default an agent uses the Claude backend (`claude-opus-4-8`), which reads
`ANTHROPIC_API_KEY` from the environment. synapse is provider-neutral — pick any
backend:

```python
from synapse import Agent, AnthropicModel, OpenAIModel
Agent("calc", model=AnthropicModel("claude-sonnet-4-6"))
Agent("calc", model=OpenAIModel("gpt-4o"))                 # or any OpenAI-compatible
Agent("calc", model=OpenAIModel("qwen2.5", base_url="http://localhost:11434/v1", api_key="x"))
```

See the [guide](guide.md#models--providers) for details.

## Offline / testing — no API key

Use the built-in backends that need no network:

```python
from synapse import Agent, ScriptedModel, EchoModel

# Replays a fixed script of turns (tool calls + final text) — ideal for tests.
agent = Agent("calc", model=ScriptedModel([[("add", {"a": 2, "b": 2})], "it's 4"]))

# Echoes the last user message.
agent = Agent("echo", model=EchoModel())
```

## Three ways to invoke

```python
# 1. Sync — a thin wrapper, works anywhere (scripts, notebooks, threads)
result = agent.run("hello")

# 2. Async — use inside an event loop; many runs concurrently
result = await agent.arun("hello")
results = await asyncio.gather(agent.arun(a), agent.arun(b))   # parallel

# 3. Streaming — events as they happen
async for ev in agent.astream("hello"):
    if ev.type == "text_delta":
        print(ev.text, end="", flush=True)
    elif ev.type == "run_complete":
        print(ev.result.output)
```

`RunResult` carries: `output`, `messages`, `iterations`, `stop_reason`,
`usage` (tokens), `verify_rounds`.

## Composing agents

Any agent can be a tool for another agent:

```python
researcher = Agent("researcher", tools=[search])
writer = Agent("writer", instructions="Summarize crisply.")
boss = Agent("boss", tools=[researcher.as_tool(), writer.as_tool()])
boss.run("Explain X, briefly.")   # delegations in one turn run concurrently
```

Next: the [Guide](guide.md) for the full option surface.
