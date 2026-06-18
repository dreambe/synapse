# Changelog

All notable changes to synapse are documented here. This project is
pre-1.0 and experimental; the API may change between minor versions.

## [0.3.0] — unreleased

Focus of this release: **depth and honesty over breadth.** Deepen the three
real frontier gaps, govern concurrency, and stop overselling.

### Changed
- **Type-checked.** Added `mypy` to the dev deps and CI (it ran clean across all
  41 modules). synapse shipped a `py.typed` marker but had never been type-checked
  — closing that integrity gap. Fixes along the way: `Message`/`ToolResultBlock`
  `content` now honestly typed `str | list` (the constructor's str shorthand was
  a lie in the annotation); runtime `user_input` typed `str | list` for
  multimodal input; server host/port coerced to `str`/`int`.

### Added
- **Structured permission modes.** A `PermissionPolicy` (`permission_policy(...)`)
  raises per-tool approval to a declared, run-wide **mode** — `PLAN` (read-only:
  allow reads, deny every mutating tool), `ASK` (allow reads, escalate writes to
  the approval callback; deny if none — fail safe), `AUTO` (allow), `BYPASS`
  (allow, explicitly). A tool's new `side_effect` field (`"read"` vs the safe
  default `"write"`) drives classification; `allow`/`deny` name-sets override the
  mode. "This run is read-only" becomes one structural decision, not a per-tool
  habit. Wired via `arun(permissions=...)` / `RunContext.permissions`.
- **Governance (rate limits, quotas, concurrency) for agents-as-services.** A
  `Governor` hands out an **isolated** rate limiter + quota + concurrency cap
  **per scope** (tenant/user) — `async with governor.admit(scope)` is the one
  admission call a server makes per request. Building blocks usable alone:
  `RateLimiter` (async token bucket, burst + sustained rate), `Quota`
  (cumulative budget with optional period reset). The clock is injectable, so
  behavior is deterministic in tests; `RateLimited` / `QuotaExceeded` signal
  admission failure. Scope keys must come from authn (same rule as tenant memory).
- **Durable execution: atomic mid-batch recovery.** The journal already records
  each tool result write-ahead; now a resumed run (same `run_id` +
  `checkpointer` + `journal`) whose transcript ends at an unfinished parallel
  tool batch **re-runs that batch before the next model call** — the journal
  replays the tools that completed (no double side effects) and the rest run, so
  the batch finishes atomically. The idempotency ordinal is derived from the
  transcript, so keys stay stable across the crash; `arun_stream` no longer
  appends a spurious user turn when resuming mid-flight. Closes the "no atomic
  recovery of a half-finished parallel batch" gap.
- **Stronger compaction.** `Compactor` now also triggers on an estimated
  **token** threshold (`trigger_tokens=`), **preserves the original task** (first
  user turn) verbatim, and summarizes the middle into a *structured* recap
  (decisions / facts / open threads / artifacts) instead of a flat blob. Still
  lossy by nature (documented) — but no longer drops the task statement or fires
  only on message count.
- **Mid-run steering (interruptibility).** A run is no longer a black box you can
  only kill and restart. Pass a `Steer` channel (`steer=`); a supervisor — a
  human or another agent — can `steer.send("focus on X")` to inject guidance into
  a *running* agent, or `steer.stop()` for a graceful halt. Steering is applied at
  the next **turn boundary** (never mid-turn, so the loop stays consistent):
  injected guidance enters the transcript as a `[steering] …` user turn and
  streaming consumers receive a `Steered` event; a stop yields
  `stop_reason="steered_stop"`. Sends are plain sync calls, so you can steer from
  any context (typically: run in one task, steer from another). This mirrors a
  publicly-observable frontier-agent behavior (interrupt/steer), built on
  synapse's own streaming loop — not on any leaked source.
- **Curated default harness (the framework's spine).** `default_harness()` /
  `DEFAULT_INSTRUCTIONS` give synapse an opinionated, ready-to-use system prompt
  — agency/persistence, plan-then-act, gather-then-act context discipline,
  verify-before-done, honesty about uncertainty, act-on-sensible-defaults. It is
  **synthesized from widely-shared, publicly-discussed agent-harness design
  principles — not derived from any proprietary or leaked prompt** — and every
  clause is wired to a mechanism synapse already ships (the default also turns on
  the loop/circuit/no-progress guards and tool-input validation, so the prose
  and the runtime agree). `role=`/`extra=`/overrides compose; it stays a normal
  editable `Harness` that Self-Harness can evolve. Closes the standing "no
  default harness content" gap.
- **Evaluation standard (versioned benchmark).** `Benchmark` is a named,
  versioned suite of dimension-tagged `Probe`s; running it yields a `Scorecard`
  (per-dimension pass rates + an overall score, tagged with benchmark/synapse
  version, model, timestamp; `save`/`load` as a baseline). `compare(baseline,
  current)` diffs two scorecards and flags **regressions** per dimension and
  per case — so *iterations are comparable*, not vibes. `agent_probes(agent,
  cases)` folds an ordinary `Case` suite into a benchmark to score *your*
  agent; the bundled `synapse_benchmark()` is the framework's own regression
  spine — ten probes that assert *mechanism correctness* (tool use, loop
  guards, journal idempotency, plan tracking, context offload, structured
  output, tenant isolation, model failover, multimodal results) deterministically
  with `ScriptedModel`. Honest scope: it measures framework mechanisms, **not**
  model intelligence — a model-quality benchmark is a different suite built with
  `agent_probes`.
- **Run monitor (observability data plane + dashboard).** `Monitor` is a
  `Hooks` implementation you attach (`hooks=monitor`) that turns a run's
  lifecycle into a stream of structured `ActivityEvent`s and a live `RunView`
  per run. It mints a run id at start and tracks it in a `ContextVar`, so
  concurrent runs — and sub-agent runs that inherit the monitor — get distinct
  ids and a correct parent→child link; `Monitor.tree()` reconstructs the
  **agent tree** (who spawned whom). Tool calls are classified by synapse's own
  conventions (`ask_*` → sub-agent spawn, `load_skill` → skill, `run_python` →
  script, MCP-style names → mcp, …) so the feed reads semantically.
  `agent.as_tool(hooks=monitor)` forwards the monitor so you can see *inside* a
  delegated sub-agent. `monitor_app(monitor)` is a minimal dependency-free ASGI
  dashboard (`GET /` live page, `/api/runs` tree, `/api/stream` SSE). The
  framework owns the data plane + a reference page; a production console is an
  app concern on the same data.
- **Long-horizon execution (first-principles fixes).**
  - *Explicit plan* — `plan=True` gives the agent a maintained to-do list
    (`write_plan`/`update_step`), rendered into context every turn and surfaced
    on `RunResult.plan`. Decomposition becomes a tracked artifact, not a hope.
  - *Context offloading* — `offload_over=` moves large tool results to a
    `ResultStore`, leaving a preview + reference in context; the agent pulls the
    full/filtered result via the `fetch_result` tool. Context is treated as the
    scarce resource it is.
  - *Outcome verification* — `command_verifier([...])` makes `verify=` run a
    real check (tests/build); failures feed back and the agent iterates.
  - *Idempotent execution* — `journal=` + `run_id` replay recorded tool results
    instead of re-firing side effects on a retry/resume (`ExecutionJournal`,
    in-memory/file).
  - *Run records* — `RunRecorder` + `RunStore` persist the full trajectory
    (input/output/usage/messages/stop reason) as a queryable fact source.
- **Multi-tenant isolation.** `MemoryNamespace` (`InMemoryNamespace` /
  `FileNamespace` / `VectorNamespace`) hands out an **isolated `Memory` per
  scope** (tenant/user) — `ns.scope("alice")` can never recall
  `ns.scope("bob")`'s facts; `FileNamespace` sanitizes the scope key (no path
  traversal). Closes a real leak: a single shared memory previously let one
  user's `recall` see another's. `Session` gained a `scope` field; conversation
  isolation is distinct `Session` objects per (tenant, conversation).
- **Enterprise knowledge grounding + human escalation.** `grounding=` runs a
  preflight provider (e.g. an MCP-backed knowledge graph) before the loop and
  injects the result as business context, so the agent knows *which repo /
  scope / red lines* while decomposing the task; the KG's own MCP tools cover
  ad-hoc mid-task questions. `human_tool(channel)` + `HumanChannel` /
  `CallbackChannel` let the agent escalate to a person (e.g. a Feishu bot) when
  blocked, with the reply flowing back into the run.
- **Self-Harness (experimental).** The harness becomes a versioned, declared
  object (`Harness`) and improves itself through a regression-gated loop
  (`evolve`): weakness mining over eval failures (`mine_weaknesses` /
  `classify_failure`, reusing loop-engineering `stop_reason`s) → bounded, audited
  edits to *declared* surfaces only (`HarnessEdit`; `model_proposer` emits them
  as validated JSON) → promotion only if ≥1 split improves and **neither
  regresses** (held-in / held-out via the eval harness). `aevaluate`/`evaluate`
  gained `run_kwargs`; `CaseResult` now carries `stop_reason`.
- **Code execution sandbox.** `code_execution_tool()` / `run_python()` run code
  in an isolated subprocess (fresh temp dir, minimal env, POSIX CPU/memory/output
  limits, hard timeout). Explicitly process isolation, **not** a security
  boundary against adversarial code (no syscall/network isolation) — documented.
- **Semantic memory.** `VectorMemory` with a pluggable `Embedder` (cosine
  search); `HashingEmbedder` (offline, dep-free) and `OpenAIEmbedder` (`[openai]`).
- **Native OpenAI streaming.** `OpenAIModel.stream` now streams tokens (and
  accumulates streamed tool calls) instead of returning a single chunk.
- **A2A task persistence.** Pluggable `TaskStore` — `InMemoryTaskStore`
  (default) and `FileTaskStore` (survives restarts); `A2ADispatcher(task_store=)`.
- **Loop control (loop engineering).** Stopping guards beyond the iteration cap:
  `max_repeated_tool_calls` (loop detection — same call recurring across turns →
  `loop_detected`), `max_consecutive_tool_errors` (circuit breaker →
  `tool_errors_exhausted`), and `max_no_progress` (same tool-call set repeating →
  `no_progress`). Each stops the run gracefully with a descriptive `stop_reason`;
  off by default, and loop detection counts per-turn so parallel fan-out isn't
  falsely flagged.
- **Structured outputs.** Give a run an `output_schema` (JSON Schema) or a
  `response_model` (Pydantic); the schema is described to the model and the
  final answer is parsed + validated into `RunResult.parsed` (provider-neutral,
  composes with `verify=`). Plus opt-in `validate_tool_inputs=` — a malformed
  tool call is returned to the model as a correctable error instead of raising.
- **Eval harness.** `Case` / `evaluate` / `aevaluate` / `Report` with checks
  (`contains`, `equals`, `matches`, arbitrary predicates, and `llm_judge`
  against a rubric) so agent behavior is measurable and changes are verifiable.
- **Tracing.** `TracingHooks` (in-memory spans, dependency-free) and `OTelHooks`
  (OpenTelemetry, `[otel]` extra) built on the lifecycle hooks.
- **Typed package + live tests.** Ships a `py.typed` marker; gated live-model
  integration tests (`pytest -m integration`, self-skipping without keys) plus a
  CI job that runs them when API-key secrets are present.
- **Skills (progressive disclosure).** A skill is a folder (`SKILL.md` front
  matter + instructions + optional bundled files). `load_skills(dir)` /
  `Skill.from_directory` discover them; `Agent(skills=...)` puts each skill's
  *description* in context and adds `load_skill` / `read_skill_file` tools so
  the model loads full instructions and bundled resources (text/image/document)
  only when relevant. Path-traversal-guarded.
- **Provider-neutral backends.** Added `OpenAIModel` — works with any OpenAI
  Chat Completions-compatible endpoint (OpenAI, Azure, Together, Groq, Ollama,
  vLLM, …) via `base_url`. Proves the `Model` abstraction; Anthropic stays the
  default. `RetryModel` fallbacks work across providers.
- **Multimodal content.** `ImageBlock` and `DocumentBlock` (PDF / text, via
  base64 / url / file / inline text) can appear in user input and tool results;
  tool results may return rich content instead of only a string. Backends
  serialize them natively (Anthropic) or by best effort (OpenAI). Incoming A2A
  `FilePart`s map to image or document input.
- **Token usage on A2A tasks.** Completed A2A tasks expose token usage in
  `metadata`. (Dollar-cost pricing was prototyped and removed — usage only.)
- **Documentation.** A `docs/` set: getting-started, full guide (every run
  option, incl. multimodal), and A2A integration.
- **A2A protocol compliance (v0.3.0).** A strict Agent2Agent implementation:
  `spec` types (Message/Part/Task/TaskStatus/Artifact/AgentCard with the exact
  `kind` discriminators and `TaskState` literals), a JSON-RPC 2.0
  `A2ADispatcher` (`message/send`, `message/stream` SSE, `tasks/get`,
  `tasks/cancel`, `tasks/resubscribe`, `tasks/pushNotificationConfig/set`+`get`),
  the compliant `create_a2a_app` server (card at `/.well-known/agent-card.json`),
  and an `A2AClient` that talks to *any* A2A agent. **Surpass:** `A2AClient.as_tool()`
  wraps a standards-compliant remote agent as a synapse tool (cross-ecosystem
  delegation), plus **signed, retrying push-notification webhooks**
  (`synapse.a2a.push`: HMAC-SHA256 over `timestamp + "." + body` with replay
  protection via `verify_signature`, exponential-backoff retries). The earlier
  synapse-native `/run` server is kept as a labelled convenience layer.
- **Streaming as the execution primitive.** `Model.stream()` (native token
  streaming in the Anthropic backend; a single-chunk default for every other
  backend), `Agent.astream()` yielding `TextDelta` / `ToolCall` / `ToolOutput`
  / `RunComplete`. `run`/`arun` consume the same stream — one loop, one source
  of truth.
- **A2A streaming** over SSE: `POST /run/stream` on both the stdlib and ASGI
  servers, and `RemoteAgent.astream(...)` on the client.
- **Concurrency & timeout governance:** `max_parallel_tools` (bounded fan-out
  via a semaphore), `timeout` (wall-clock, raises `RunTimeout`), `tool_timeout`
  (per-tool, surfaced as a tool error).
- **Transient vs permanent errors:** `RetryModel` gained `retry_on`; a
  non-transient exception (e.g. a bug) is no longer retried or swallowed.

### Changed
- **`RunContext` is now the canonical configuration object.** `Agent.run` is a
  thin sync forwarder to `arun`; the giant duplicated signature is gone.
- Tool execution errors now include the exception type, instead of an opaque
  string, so failures are diagnosable.
- README reframed as **experimental**; overclaims ("production", "durable
  execution", unqualified "A2A") downgraded to accurate language; a "Status &
  honest limitations" section added.

### Fixed
- Removed a shared-mutable-state footgun (verify-round count was smuggled via a
  function attribute, unsafe under concurrent runs); it now rides the internal
  turn sentinel.

## [0.2.0]

### Added
Observability hooks + token usage, tool approval (HITL), verifier loop,
guardrails, cross-run memory, routing, `RetryModel`, token budgets, context
compaction, tool search, checkpointing, MCP adapter, teams + blackboard.

## [0.1.0]

Initial framework: async-first `Agent`, `@tool`, provider-neutral messages,
pluggable model backends (Anthropic / Echo / Scripted), `AgentRegistry`, CLI,
and agent-to-agent (in-process, plus stdlib + ASGI HTTP servers and a client).
