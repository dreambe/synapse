# synapse

**A composable agent framework. Build an agent once — call it anywhere.**

> ⚠️ **Status: experimental (v0.3), zero production mileage.** The core loop,
> async concurrency, streaming, and A2A transport are solid and tested; several
> capabilities (keyword-based memory/tool-search/routing, checkpoint-resume,
> teams) are deliberate *first cuts*, labelled as such below. Don't ship this to
> production yet. See [Status & honest limitations](#status--honest-limitations).

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
pip install "synapse[openai]"       # + OpenAI-compatible backend (OpenAI/Azure/Together/Ollama/vLLM…)
pip install "synapse[server]"       # + uvicorn for async high-concurrency serving
pip install "synapse[mcp]"          # + Model Context Protocol client
```

Requires Python 3.10+.

## Documentation

Full guides live in [`docs/`](docs/index.md): [getting started](docs/getting-started.md) ·
[guide](docs/guide.md) (all run options) · [agent-to-agent](docs/a2a.md).

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

## Agent-to-agent (A2A)

synapse speaks the **Agent2Agent protocol v0.3.0** (JSON-RPC 2.0), so it
interoperates with any A2A-compliant peer — and a compliant remote agent can be
wrapped as a local tool, bridging ecosystems.

### Serve an agent to the A2A ecosystem

```python
from synapse.a2a import create_a2a_app
app = create_a2a_app(agent)          # uvicorn yourmodule:app
```

Endpoints (A2A-compliant):

| Method | Path                              | Purpose                                  |
|--------|-----------------------------------|------------------------------------------|
| `GET`  | `/.well-known/agent-card.json`    | Agent Card (discovery)                   |
| `POST` | `/`                               | JSON-RPC 2.0                             |

JSON-RPC methods: `message/send`, `message/stream` (SSE), `tasks/get`,
`tasks/cancel`, `tasks/resubscribe`, `tasks/pushNotificationConfig/set`+`get`.
Tasks carry the real lifecycle (`submitted → working → completed/canceled/…`)
with artifacts and history.

### Call any A2A agent — and wrap it as a tool

```python
from synapse.a2a import A2AClient

client = A2AClient("http://some-a2a-agent.example.com")
print(client.card().name)                 # discover (works for any A2A agent)
print(client.ask("summarize the news"))   # message/send → final text
async for ev in client.astream_message("..."):  # message/stream (SSE)
    ...

# Cross-ecosystem delegation: a synapse agent calls a standards-compliant peer.
coordinator = Agent("boss", tools=[client.as_tool()])
```

### synapse-native convenience layer

For quick local/dev use there's also a lightweight `/run` + `/run/stream`
server (not the A2A wire format — a synapse convenience):

```python
from synapse.a2a import serve, RemoteAgent
serve(agent, port=8080)                       # stdlib threaded, zero-dep
RemoteAgent("http://127.0.0.1:8080").run("hi")
```

```bash
synapse serve examples/basic_agent.py:agent --port 8080
```

## CLI

```bash
synapse run   mymod:agent "your task here"      # run once
synapse run   mymod:agent "task" --json         # full result as JSON
synapse serve mymod:agent --port 8080           # serve over HTTP
synapse card  http://localhost:8080             # fetch a remote agent card
```

Targets are `module:attribute` or `path/to/file.py:attribute`.

## Capabilities

The canonical configuration object is **`RunContext`**; the keyword arguments on
`run`/`arun`/`astream` are sugar over it, so there's one option surface. They
map onto the canonical agentic design patterns while tracking where the frontier
has moved (streaming, verifier loops, context engineering, observability).
Maturity is marked honestly — *solid* vs *first cut*.

```python
result = agent.run(
    "ship the release",
    hooks=CollectingHooks(),            # observability: lifecycle events + token usage
    approval=lambda name, inp: ...,     # human-in-the-loop: gate tools (requires_approval=True)
    verify=my_checker,                  # reflection → iterate-until-pass (returns Verdict)
    input_guardrails=[block_keywords([...])],
    output_guardrails=[redact(r"...")], # guardrails on input/output
    token_budget=200_000,               # resource-aware: stop when the budget is hit
    checkpointer=FileCheckpointer("runs"), run_id="job-42",  # durable execution / resume
)
```

| Capability | API | Maturity |
|---|---|---|
| Structured outputs | `output_schema=` / `response_model=` → `RunResult.parsed` | **solid** |
| Tool-input validation | `validate_tool_inputs=True` | **solid** |
| Eval harness | `Case`, `evaluate`, `contains`/`llm_judge`/… | **solid** |
| Tracing | `TracingHooks` (in-memory), `OTelHooks` (OpenTelemetry) | **solid** |
| Streaming | `agent.astream(...)`, `Model.stream` | **solid** |
| Multimodal (images & documents, in & out) | `ImageBlock`, `DocumentBlock` | **solid** |
| Provider-neutral backends | `AnthropicModel`, `OpenAIModel` (+ any OpenAI-compatible) | **solid** |
| Bounded tool concurrency | `max_parallel_tools=` | **solid** |
| Wall-clock + per-tool timeouts | `timeout=`, `tool_timeout=` → `RunTimeout` | **solid** |
| Observability hooks + token usage | `Hooks`, `CollectingHooks`, `Usage` | **solid** |
| Tool approval (HITL) | `approval=`, `@tool(requires_approval=True)` | **solid** |
| Verifier loop (iterate-until-pass) | `verify=` → `Verdict` | **solid** |
| Guardrails | `input_guardrails=` / `output_guardrails=` | **solid** |
| Resilience (transient vs permanent) | `RetryModel(model, fallbacks=[...], retry_on=...)` | **solid** |
| Token budgets | `token_budget=` | **solid** |
| MCP tools | `synapse.mcp.tools_from_session` | **solid** |
| A2A protocol v0.3.0 (JSON-RPC, tasks, SSE) | `create_a2a_app`, `A2AClient` | **solid** |
| Skills (progressive disclosure) | `Skill`, `load_skills`, `Agent(skills=...)` | **solid** |
| Cross-run memory | `Agent(memory=...)`, `InMemoryMemory`, `FileMemory` | first cut — keyword search |
| Routing | `Router`, `ModelRouter` | first cut — `Router` keyword-based |
| Tool search | `Agent(tool_search=True)`, `select_tools` | first cut — keyword ranking |
| Compaction | `Compactor` | first cut — naive prefix summary |
| Checkpoint / resume | `Checkpointer`, `run_id=` | first cut — replays tools on resume (not idempotent) |
| Teams + blackboard | `Team`, `Blackboard` | first cut |

See [`examples/advanced_agent.py`](examples/advanced_agent.py) for several of
these together.

### Streaming

```python
async for event in agent.astream("write a poem"):
    if event.type == "text_delta":
        print(event.text, end="", flush=True)
    elif event.type == "run_complete":
        print("\n--", event.result.usage.total_tokens, "tokens")
```

Streaming is the underlying execution model — `run`/`arun` consume the same
stream to a final result, so there's one loop, not two code paths. The A2A
server exposes it at `POST /run/stream` (SSE); the client consumes it via
`RemoteAgent.astream(...)`.

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
   ┌──────────┬──────────┬───────────┐
   ▼          ▼          ▼           ▼
Anthropic  OpenAI*    Echo      Scripted
 (Claude)  (+compat) (offline)  (testing)

* OpenAI backend works with any OpenAI-compatible endpoint (Azure, Together,
  Groq, Ollama, vLLM, …) via base_url.
```

| Module                | Responsibility                                        |
|-----------------------|-------------------------------------------------------|
| `synapse.agent`       | The `Agent` class and composition (`as_tool`, `card`) |
| `synapse.runtime`     | The async agent loop, sync wrapper, `Session`, `RunResult` |
| `synapse.tool`        | The `@tool` decorator and schema generation           |
| `synapse.messages`    | Provider-neutral messages and content blocks          |
| `synapse.models`      | The `Model` interface and backends                    |
| `synapse.registry`    | `AgentRegistry` for name-based lookup/routing         |
| `synapse.router`      | `Router` / `ModelRouter` — pick an agent, then run it |
| `synapse.observability` | Hooks, lifecycle events, token `Usage`              |
| `synapse.tracing`     | `TracingHooks` (in-memory) + `OTelHooks` (OpenTelemetry) |
| `synapse.structured`  | JSON-Schema validation + structured-output parsing    |
| `synapse.evaluation`  | Eval harness: `Case` / `evaluate` / checks / `llm_judge` |
| `synapse.skill`       | Skills: `SKILL.md` folders, progressive disclosure    |
| `synapse.memory`      | Cross-run `Memory` backends + auto memory tools       |
| `synapse.guardrails`  | Input/output guardrails                               |
| `synapse.context`     | `Compactor` and tool selection (context engineering)  |
| `synapse.checkpoint`  | `Checkpointer` backends for durable execution         |
| `synapse.team`        | `Team` + `Blackboard` multi-agent collaboration       |
| `synapse.mcp`         | Adapter exposing MCP server tools as synapse tools    |
| `synapse.a2a`         | A2A protocol v0.3.0 (spec types, JSON-RPC dispatcher, compliant server + client) and the native convenience layer |

## Status & honest limitations

This is an experimental framework with a deliberate point of view, not a
finished product. What it does **not** do yet (by design or as known debt):

- **Checkpointing is not true durable execution.** It snapshots the message
  transcript; resuming an interrupted run **replays tools** (not idempotent) and
  doesn't recover a half-finished parallel tool batch atomically.
- **The keyword heuristics are placeholders.** Memory recall, tool search, and
  the rule-based `Router` use lexical overlap. Swap in embeddings/an LLM for
  anything real.
- **No distributed/observability backends.** Hooks are in-process callbacks;
  there's no OpenTelemetry/trace export yet.
- **Live-model testing is opt-in, not continuous.** The default suite is fully
  offline (fakes). Gated integration tests (`pytest -m integration`) exercise a
  real model, but only run when `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` secrets
  are configured — so day-to-day CI proves the plumbing, not a live model.
- **A2A push notifications** are HMAC-signed (`X-A2A-Signature`, verifiable via
  `synapse.a2a.push.verify_signature`) and retried with backoff, but delivery is
  still in-process (no durable outbox/queue across restarts yet).

See [CHANGELOG.md](CHANGELOG.md) for what landed when.

## Development

```bash
pip install -e ".[dev]"
pytest          # the suite runs fully offline (no API key needed)
ruff check .
```

## License

MIT — see [LICENSE](LICENSE).
