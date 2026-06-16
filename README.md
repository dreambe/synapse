# synapse

**A composable agent framework. Build an agent once — call it anywhere.**

synapse is a small, dependency-free Python framework for building LLM agents
that are meant to be *invoked*, not just chatted with. The same `Agent` can be:

- **run in-process** — `agent.run("...")` (sync) or `await agent.arun("...")`
- **composed** — exposed as a tool to another agent (in-process A2A)
- **served** — turned into an HTTP service with one call
- **called remotely** — over the agent-to-agent (A2A) protocol, including
  agent → agent delegation across processes

The execution layer is **async-first**, so a single process handles many
concurrent runs (inbound *and* outbound) without the GIL getting in the way —
agent work is I/O-bound, which is exactly what asyncio is for.

It draws on ideas from agent frameworks like Hermes and the broader
open-agent ecosystem, but the goal is different: synapse is a *library and
protocol* for agents-as-services, not a single conversational assistant.

## Why another agent framework?

Most agent libraries assume the agent *is* the application — a chat loop you
talk to. synapse treats an agent as a reusable unit of behavior with a stable
calling convention, so it slots into larger systems: microservices, pipelines,
and multi-agent meshes where agents call each other.

- **Provider-neutral core.** The framework speaks one message format
  (`synapse.messages`); LLM backends convert to and from it. Swapping Claude
  for a test double is a one-line change.
- **Zero required dependencies.** The core and the A2A transport are pure
  standard library. The Anthropic backend is an optional extra.
- **A2A is first-class.** Agent-to-agent calls work the same whether the peer
  is an object in the same process or a service across the network.

## Install

```bash
pip install synapse                 # core (stdlib only)
pip install "synapse[anthropic]"    # + the Claude backend
pip install "synapse[server]"       # + uvicorn for async high-concurrency serving
```

Requires Python 3.10+.

## Quick start

```python
from synapse import Agent, tool

@tool
def add(a: int, b: int) -> int:
    "Add two integers."
    return a + b

@tool
def multiply(a: int, b: int) -> int:
    "Multiply two integers."
    return a * b

agent = Agent(
    "calculator",
    instructions="You are a precise calculator. Use the tools for arithmetic.",
    tools=[add, multiply],
)

result = agent.run("What is 19 * 23, then minus 7?")
print(result.output)
```

By default an agent uses the Claude backend (`claude-opus-4-8`), which reads
`ANTHROPIC_API_KEY` from the environment. For tests and offline work, use the
built-in backends that need no API key:

```python
from synapse import Agent, ScriptedModel, EchoModel

# Replays a fixed sequence of turns — great for tests.
agent = Agent("calc", model=ScriptedModel([[("add", {"a": 2, "b": 2})], "it's 4"]))
```

## Tools

Decorate any function. synapse derives the JSON schema from its type hints and
the description from its docstring:

```python
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    "Get the current weather for a city."
    ...
```

Optional parameters (those with defaults) are omitted from `required`.

## Composing agents (in-process A2A)

Any agent can be handed to another agent as a tool:

```python
researcher = Agent("researcher", instructions="Find facts.", tools=[search])
writer = Agent("writer", instructions="Summarize crisply.")

coordinator = Agent(
    "coordinator",
    instructions="Delegate research, then ask the writer to summarize.",
    tools=[researcher.as_tool(), writer.as_tool()],
)

coordinator.run("Explain X, briefly.")
```

See [`examples/multi_agent.py`](examples/multi_agent.py).

## Concurrency

Agents are async under the hood, so the framework scales on a single event
loop:

- **High-concurrency serving (inbound).** Run the ASGI app under uvicorn and
  one process serves many simultaneous `/run` requests — no thread-per-request
  ceiling. The CLI does this automatically when `synapse[server]` is installed.
- **Parallel delegation (outbound).** When a model requests several tools in
  one turn, they execute concurrently via `asyncio.gather`. Fanning out to N
  sub-agents (each an `as_tool`) costs **one** round of latency, not N.
- **Safe sessions.** Distinct sessions run fully in parallel; turns on the
  *same* session are serialized by a per-session lock.

```python
import asyncio
from synapse import Agent

# Run many agents concurrently
results = await asyncio.gather(*(a.arun(task) for a, task in jobs))
```

Use `await agent.arun(...)` inside async code; `agent.run(...)` is a thin sync
wrapper that works anywhere (scripts, notebooks, threads).

## Serving an agent (network A2A)

Turn an agent into an HTTP service. Two paths, same protocol:

```python
# Zero dependencies — stdlib threaded server (great for dev/tests)
from synapse.a2a import serve
serve(agent, port=8080)

# High concurrency — native-async ASGI app under uvicorn
from synapse.a2a import create_app
app = create_app(agent)          # run with: uvicorn yourmodule:app
```

```bash
synapse serve examples/basic_agent.py:agent --port 8080
# uses uvicorn automatically if synapse[server] is installed, else stdlib
```

Endpoints:

| Method | Path                        | Purpose                          |
|--------|-----------------------------|----------------------------------|
| `GET`  | `/.well-known/agent.json`   | Agent card (discovery)           |
| `GET`  | `/health`                   | Liveness probe                   |
| `POST` | `/run`                      | Run the agent on a task          |

Call it from another process:

```python
from synapse.a2a import RemoteAgent

remote = RemoteAgent("http://127.0.0.1:8080")
print(remote.card().skills)            # discover capabilities
print(remote.run("hello").output)      # invoke it (sync)
await remote.arun("hello")             # or async — many calls in flight at once

# Or let a local agent delegate to the remote one, over the wire:
local_coordinator = Agent("boss", tools=[remote.as_tool()])
```

## CLI

```bash
synapse run   mymod:agent "your task here"      # run once
synapse run   mymod:agent "task" --json         # full result as JSON
synapse serve mymod:agent --port 8080           # serve over HTTP
synapse card  http://localhost:8080             # fetch a remote agent card
```

Targets are `module:attribute` or `path/to/file.py:attribute`.

## Architecture

```
            ┌──────────────────────────────────────────┐
            │                  Agent                    │
            │   instructions · tools · model · card     │
            └───────────────┬───────────────────────────┘
                            │ run_agent() — the loop
              ┌─────────────┼──────────────┐
              ▼             ▼              ▼
          Model         Tools         A2A transport
        (backends)   (@tool fns)   (server · client)
              │
   ┌──────────┼───────────┐
   ▼          ▼           ▼
Anthropic   Echo      Scripted
 (Claude)  (offline)  (testing)
```

| Module                | Responsibility                                        |
|-----------------------|-------------------------------------------------------|
| `synapse.agent`       | The `Agent` class and composition (`as_tool`, `card`) |
| `synapse.runtime`     | The async agent loop, sync wrapper, `Session`, `RunResult` |
| `synapse.tool`        | The `@tool` decorator and schema generation           |
| `synapse.messages`    | Provider-neutral messages and content blocks          |
| `synapse.models`      | The `Model` interface and backends                    |
| `synapse.registry`    | `AgentRegistry` for name-based lookup/routing         |
| `synapse.a2a`         | Agent-to-agent protocol, stdlib + ASGI servers, client |

## Development

```bash
pip install -e ".[dev]"
pytest          # the suite runs fully offline (no API key needed)
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
