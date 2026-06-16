# Guide

Everything beyond a bare `agent.run(text)` is opt-in. This page is the full
option surface.

## Tools

Decorate any function (sync or `async def`). The JSON schema is derived from
type hints; the description from the docstring.

```python
@tool
def get_weather(city: str, unit: str = "celsius") -> str:
    "Get the current weather for a city."
    ...
```

Parameters with defaults are optional. Gate a sensitive tool behind approval:

```python
@tool(requires_approval=True)
def deploy(service: str) -> str:
    "Deploy to production."
    ...
```

## RunContext

`RunContext` is the **canonical configuration object**; the keyword arguments on
`run`/`arun`/`astream` are sugar over it. Pass either form:

```python
from synapse import RunContext
ctx = RunContext(max_iterations=8, token_budget=200_000, timeout=60)
agent.run("...", context=ctx)
# equivalent:
agent.run("...", max_iterations=8, token_budget=200_000, timeout=60)
```

| Option | Meaning |
|---|---|
| `max_iterations` | Max model↔tool turns (default 12). |
| `hooks` | Lifecycle/observability callbacks (see below). |
| `approval` | `(tool_name, input) -> bool \| ApprovalDecision` for `requires_approval` tools. |
| `verify` | `(output) -> bool \| Verdict`; iterate-until-pass. |
| `max_verify_rounds` | Verifier retry budget (default 3). |
| `token_budget` | Stop the run once cumulative tokens reach this. |
| `timeout` | Wall-clock seconds; raises `RunTimeout`. |
| `tool_timeout` | Per-tool seconds; a timeout becomes a tool error. |
| `max_parallel_tools` | Bound concurrent tool execution in a turn. |
| `input_guardrails` / `output_guardrails` | Validate/transform text. |
| `compactor` | Summarize old turns when history grows (see Context). |
| `checkpointer` + `run_id` | Persist/resume run state. |
| `pricing` | Override the cost price table. |
| `tool_search` | Expose a `search_tools` meta-tool, load schemas on demand. |

## Observability & hooks

```python
from synapse import CollectingHooks

hooks = CollectingHooks()          # records every event
result = await agent.arun("...", hooks=hooks)
print(result.usage.total_tokens)   # tokens
print([e[0] for e in hooks.events])

# Subclass Hooks to integrate your own tracing:
from synapse import Hooks
class MyHooks(Hooks):
    async def on_tool_start(self, name, tool_input): ...
    async def on_run_end(self, result): ...
```

Events: `on_run_start`, `on_model_response`, `on_tool_start`, `on_tool_end`,
`on_turn_end`, `on_run_end`. Sync or async methods both work.

## Human-in-the-loop (approval)

```python
def approve(tool_name, tool_input):
    return tool_name != "deploy"   # or return ApprovalDecision(False, "needs ticket")

agent.run("...", approval=approve)   # denied calls return an error to the model
```

## Verifier (iterate until it passes)

```python
from synapse import Verdict
def verify(output: str) -> Verdict:
    return Verdict(passed="DONE" in output, feedback="end with DONE")

r = agent.run("...", verify=verify, max_verify_rounds=3)
r.stop_reason   # "verified" | "verification_failed"
```

## Guardrails

```python
from synapse import block_keywords, redact, max_length
agent.run(
    "...",
    input_guardrails=[block_keywords(["rm -rf"]), max_length(10_000)],
    output_guardrails=[redact(r"\b\d{16}\b")],   # mask card numbers
)
```
Input guardrails run before the loop (a violation raises `GuardrailViolation`);
output guardrails transform/validate the final answer.

## Memory (cross-run)

`Session` is per-conversation; a `Memory` persists across runs. Attaching one
auto-adds `remember`/`recall` tools the model can drive.

```python
from synapse import Agent, InMemoryMemory, FileMemory
agent = Agent("assistant", memory=FileMemory("memory.jsonl"))
```

> First cut: recall is keyword/lexical. Implement the `Memory` ABC over an
> embedding store for semantic recall.

## Context compaction

When history grows, summarize old turns so the loop can keep going:

```python
from synapse import Compactor
compactor = Compactor(agent.model, trigger_messages=24, keep_recent=8)
agent.run("...", compactor=compactor)
```

> First cut: naive prefix summarization. Good enough to stay in-window; not a
> token-exact policy.

## Resilience

```python
from synapse import RetryModel, AnthropicModel
model = RetryModel(
    AnthropicModel("claude-opus-4-8"),
    fallbacks=[AnthropicModel("claude-sonnet-4-6")],
    max_retries=2,
)
```
Transient errors (connection/timeout/model errors) are retried then fall back;
a permanent error (a bug) is raised immediately, not retried.

## Routing

```python
from synapse import Router
router = Router(default=support_agent)
router.add_keyword_route(["invoice", "payment"], billing_agent)
router.run("question about my invoice")     # → billing_agent
```
For intent-based routing use `ModelRouter` (an LLM picks the agent).

## Teams

```python
from synapse import Team
team = Team(coordinator, [researcher, writer])   # members become tools + shared Blackboard
team.run("produce a report")
```

## Checkpoint & resume

```python
from synapse import FileCheckpointer
cp = FileCheckpointer("runs")
agent.run("step 1", checkpointer=cp, run_id="job-42")
# later / after a crash — prior history is restored, new input appended:
agent.run("step 2", checkpointer=cp, run_id="job-42")
```

> First cut: snapshots the transcript; resuming replays tools (not idempotent).

See also: [Cost accounting](cost.md) and [A2A](a2a.md).
