# Observability & evaluation

Two primitives for operating and improving an agent over time: a **run
monitor** (see what many agents are doing, live) and a **benchmark** (a
versioned, comparable evaluation standard).

## Run monitor

`Monitor` is a `Hooks` implementation. Attach it to a run and it turns the
run's lifecycle into structured `ActivityEvent`s and a live `RunView` per run.

```python
from synapse import Agent, Monitor, ScriptedModel

monitor = Monitor()
agent = Agent("calc", model=ScriptedModel(["done"]))
await agent.arun("go", hooks=monitor)

monitor.snapshot()   # every run, newest first
monitor.tree()       # root runs with sub-agent runs nested under "children"
```

### The agent tree

The monitor mints a run id at start and tracks it in a `ContextVar`. Because
asyncio tasks copy their context, **concurrent runs get distinct ids**, and a
sub-agent run that inherits the monitor is linked as a **child** of the run
that spawned it. Forward the monitor when exposing an agent as a tool:

```python
coordinator = Agent(
    "coordinator",
    tools=[researcher.as_tool(hooks=monitor)],  # see *inside* the sub-agent
    model=...,
)
await coordinator.arun("delegate this", hooks=monitor)

monitor.tree()
# [ coordinator → children: [ researcher ] ]
```

### Semantic activity

Tool calls are classified by synapse's own naming conventions, so the feed
reads as intent, not a flat log:

| tool name             | kind       |
|-----------------------|------------|
| `ask_*`               | `subagent` |
| `load_skill`, `read_skill_file` | `skill` |
| `run_python`          | `script`   |
| `write_plan`, `update_step` | `plan` |
| `fetch_result`        | `fetch`    |
| `remember`, `recall`  | `memory`   |
| MCP-style (`pkg.tool`)| `mcp`      |

### Live dashboard

`monitor_app(monitor)` is a minimal, dependency-free ASGI dashboard:

```python
from synapse import Monitor, monitor_app

monitor = Monitor()
app = monitor_app(monitor)   # uvicorn yourmodule:app   (pip install synapse[server])
```

- `GET /` — a self-contained HTML page: live agent tree + activity feed.
- `GET /api/runs` — the run tree as JSON.
- `GET /api/stream` — Server-Sent Events of every activity event.

The framework owns the **data plane** (events, views, subscribe stream) plus a
reference page. A production console is an app concern built on the same data —
or subscribe a queue yourself with `monitor.subscribe()`.

## Benchmark — a versioned evaluation standard

`evaluation.py` answers "did this run pass its checks?". A `Benchmark` answers
"**is this iteration better or worse than the last?**" across many capabilities,
with a persisted baseline.

```python
from synapse import synapse_benchmark, compare, Scorecard

card = synapse_benchmark().run(model="claude-...")
print(card.summary())
card.save("baseline.json")

# next iteration:
new = synapse_benchmark().run()
diff = compare(Scorecard.load("baseline.json"), new)
print(diff.summary())
assert not diff.regressed         # gate CI / self-harness promotion on this
```

A `Scorecard` carries per-dimension pass rates and an overall score, tagged
with the benchmark version, the synapse version, the model, and a timestamp.
`compare` flags **regressions** per dimension and per case (`newly_failing` /
`newly_passing`), and `diff.regressed` is the single boolean to gate on.

### Scoring *your* agent

Fold an ordinary `Case` suite into a benchmark with `agent_probes`, grouping by
dimension, to track your agent's quality across iterations:

```python
from synapse import Benchmark, Case, agent_probes, contains

probes = agent_probes(my_agent, [
    Case("2+2?", check=contains("4")),
    Case("capital of France?", expect_contains="Paris"),
], dimension="qa")
card = Benchmark("my-agent", "2025.06", probes).run()
```

### The bundled suite

`synapse_benchmark()` is the framework's own regression spine: ten probes that
assert *mechanism correctness* — tool use, loop guards, journal idempotency,
plan tracking, context offload, structured output, tenant isolation, model
failover, multimodal results — deterministically with `ScriptedModel`.

**Honest scope:** it measures framework mechanisms, **not** model intelligence.
A model-quality benchmark is a separate suite you build with `agent_probes`
over real cases against a live model.
