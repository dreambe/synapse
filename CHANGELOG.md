# Changelog

All notable changes to synapse are documented here. This project is
pre-1.0 and experimental; the API may change between minor versions.

## [0.3.0] — unreleased

Focus of this release: **depth and honesty over breadth.** Deepen the three
real frontier gaps, govern concurrency, and stop overselling.

### Added
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
