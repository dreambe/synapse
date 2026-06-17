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

## Loop control (loop engineering)

Treat the agent loop as an engineering artifact: design how it *stops*. Beyond
the hard iteration cap (`max_iterations`), wall-clock `timeout`, and
`token_budget`, three guards catch the classic runaway modes — each stops the
run gracefully with a descriptive `stop_reason` (the partial work is in
`result.messages`):

```python
result = agent.run(
    task,
    max_iterations=30,
    max_repeated_tool_calls=3,       # same call across >3 turns → "loop_detected"
    max_consecutive_tool_errors=3,   # 3 all-failing turns in a row → "tool_errors_exhausted"
    max_no_progress=3,               # same tool-call set 3 turns running → "no_progress"
    timeout=120, token_budget=200_000,
)
if result.stop_reason != "end_turn":
    ...  # hit a guard — inspect and decide
```

These are **off by default** (the iteration cap is the baseline). Loop
detection counts per *turn*, so legitimate parallel fan-out of identical calls
in one turn does not trip it. For a verifiable "done" condition (not just "stop
looping"), combine with `verify=`.

## Long-horizon execution

Treats context as the scarce resource and makes "done" mean *verified*.

**Explicit plan** — give the agent a maintained to-do list (a context anchor +
progress ledger), edited via `write_plan`/`update_step` and rendered into context
every turn:

```python
result = agent.run("ship the feature", plan=True)
result.plan        # the final Plan (steps + statuses)
```

**Offload large tool results** — don't dump a 50KB result into context. Over a
threshold, the result is stored and replaced with a short preview + a reference;
the agent pulls the full (or filtered) result on demand via `fetch_result`:

```python
agent.run("analyze the logs", offload_over=2000)         # in-memory store by default
agent.run("...", offload_over=2000, result_store=FileResultStore("blobs"))
```

**Outcome verification** — verify by *running a check* (tests/build), not by
reading the final text; failures feed back and the agent iterates:

```python
from synapse import command_verifier
agent.run("fix the bug", verify=command_verifier(["pytest", "-q"]), max_verify_rounds=3)
# stop_reason == "verified" only if the command exits 0
```

**Idempotent execution** — with a journal + `run_id`, a replay (after a crash)
returns recorded tool results instead of re-firing side effects:

```python
from synapse import FileJournal
agent.run("send the invoices", journal=FileJournal("runs/job-42.json"), run_id="job-42")
```

**Run records (fact source)** — persist the full trajectory for audit, debugging,
and Self-Harness weakness mining:

```python
from synapse import RunRecorder, FileRunStore
store = FileRunStore("runs")
agent.run("...", hooks=RunRecorder(store))
await store.list()        # input, output, stop_reason, usage, full messages
```

## Structured outputs

Get typed, validated results — not just free text — so an agent can feed
downstream systems.

```python
schema = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "country": {"type": "string"}},
    "required": ["city", "country"],
}
result = agent.run("The capital of France.", output_schema=schema)
result.parsed     # {"city": "Paris", "country": "France"}  (None if invalid)

# Or a Pydantic model — result.parsed is an instance:
from pydantic import BaseModel
class Place(BaseModel):
    city: str
    country: str
result = agent.run("...", response_model=Place)
```

The schema is described to the model and the final answer is parsed +
validated (provider-neutral). Combine with `verify=` to iterate until valid.

**Tool-input validation** — validate a tool call's arguments against the tool's
own schema before executing, so a malformed call is returned to the model as a
correctable error instead of raising:

```python
agent.run("...", validate_tool_inputs=True)
```

## Multimodal (images & documents)

Pass images and documents (PDF, text) as input, and let tools return them.

```python
from synapse import ImageBlock, DocumentBlock, TextBlock

# Multimodal input — pass a list of content blocks instead of a string:
agent.run([
    TextBlock("Summarize this and describe the chart."),
    DocumentBlock.from_file("report.pdf"),        # or .from_base64/.from_url/.from_text
    ImageBlock.from_file("chart.png"),            # or .from_base64(data, "image/png") / .from_url(...)
])

# A tool can return an image or document (or a list of blocks):
@tool
def render_chart(spec: str) -> ImageBlock:
    "Render a chart and return it as a PNG."
    return ImageBlock.from_base64(png_b64, "image/png")
```

Content serializes to the model's native blocks. Over A2A, an incoming
`FilePart` maps to an `ImageBlock` (image mime types) or a `DocumentBlock`
(everything else) automatically. (Input guardrails apply to string input; for
block-list input they're skipped.)

## Models & providers

synapse is **provider-neutral**: the framework speaks its own message format and
each backend translates. Pick a backend per agent:

```python
from synapse import Agent, AnthropicModel, OpenAIModel

Agent("a", model=AnthropicModel("claude-opus-4-8"))           # default
Agent("b", model=OpenAIModel("gpt-4o"))                       # OpenAI
# Any OpenAI-compatible endpoint via base_url: Azure, Together, Groq, Ollama, vLLM…
Agent("c", model=OpenAIModel("llama-3.1-70b",
                             base_url="http://localhost:11434/v1", api_key="x"))
```

Resilience and fallback work across providers:

```python
from synapse import RetryModel
model = RetryModel(OpenAIModel("gpt-4o"), fallbacks=[AnthropicModel("claude-opus-4-8")])
```

Both backends stream natively at the token level (`agent.astream(...)` /
`Model.stream`). To support a provider with no OpenAI-compatible endpoint,
implement the `Model` interface (`async def generate`, optional `stream`).

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
| `max_repeated_tool_calls` | Stop if one tool call recurs across more than N turns (`loop_detected`). |
| `max_consecutive_tool_errors` | Circuit breaker: stop after N all-failing turns (`tool_errors_exhausted`). |
| `max_no_progress` | Stop if the same tool-call set repeats N turns running (`no_progress`). |
| `input_guardrails` / `output_guardrails` | Validate/transform text. |
| `compactor` | Summarize old turns when history grows (see Context). |
| `checkpointer` + `run_id` | Persist/resume run state. |
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

### Tracing

`TracingHooks` records in-memory spans (run + each tool) with no dependency;
`OTelHooks` emits OpenTelemetry spans (`pip install synapse[otel]`).

```python
from synapse import TracingHooks
tracer = TracingHooks()
agent.run("...", hooks=tracer)
for span in tracer.spans:
    print(span.name, span.duration, span.attributes)
```

## Evaluation

Measure agent behavior so changes are verifiable.

```python
from synapse import Case, evaluate, contains, llm_judge

report = evaluate(agent, [
    Case("2 + 2?", check=contains("4")),
    Case("capital of France?", expect_contains="Paris"),
    Case("explain recursion", check=llm_judge(judge_model, "Mentions a base case.")),
])
print(report.summary())          # per-case PASS/FAIL + pass-rate + tokens
assert report.pass_rate >= 0.9
```

Checks: `contains` / `equals` / `matches`, any predicate `(RunResult) -> bool |
(bool, detail)`, or `llm_judge(model, rubric)` (PASS/FAIL). Checks may be async.

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

## Skills

A *skill* is a folder of packaged expertise loaded on demand. Layout:

```
skills/
  fill-pdf/
    SKILL.md          # front matter (name, description) + instructions
    template.pdf      # bundled resources (optional)
```

```markdown
---
name: fill-pdf
description: Fill out PDF forms from structured data.
---
To fill a form: read template.pdf, map fields to the data, then ...
```

Attach skills to an agent:

```python
from synapse import Agent, load_skills
agent = Agent("assistant", skills=load_skills("skills"))
```

**Progressive disclosure** — only each skill's *description* sits in the agent's
context by default (cheap). The agent loads a skill's full instructions on
demand by calling the `load_skill(name)` tool, and fetches bundled files with
`read_skill_file(skill, filename)` (which returns text, an `ImageBlock`, or a
`DocumentBlock` — so binary resources flow in multimodally). Both tools are
added automatically when `skills=` is set. Build skills in code too:

```python
from synapse import Skill
Skill(name="sql", description="Write safe SQL.", instructions="Always parametrize ...")
```

## Memory (cross-run)

`Session` is per-conversation; a `Memory` persists across runs. Attaching one
auto-adds `remember`/`recall` tools the model can drive.

```python
from synapse import Agent, InMemoryMemory, FileMemory
agent = Agent("assistant", memory=FileMemory("memory.jsonl"))
```

`InMemoryMemory` / `FileMemory` use keyword search. For **semantic recall**, use
`VectorMemory` with an embedder (cosine similarity):

```python
from synapse import VectorMemory, OpenAIEmbedder, HashingEmbedder
agent = Agent("assistant", memory=VectorMemory(OpenAIEmbedder()))   # semantic
agent = Agent("assistant", memory=VectorMemory(HashingEmbedder()))  # offline, dep-free
```

Implement the `Embedder` interface to plug in any embedding model.

## Code execution

Give an agent a sandboxed Python tool:

```python
from synapse import Agent, code_execution_tool
agent = Agent("coder", tools=[code_execution_tool(timeout=10, memory_mb=512)])
```

It runs each snippet in an isolated subprocess (fresh temp dir, minimal env,
CPU/memory/output limits, hard timeout). **Honest scope:** this is process
isolation, *not* a security boundary against adversarial code — no syscall
filtering or network isolation. For untrusted code, run inside a container /
gVisor / firejail / VM. Use `run_python(code, ...)` directly for the raw result.

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

See also: [A2A](a2a.md).
