# CLAUDE.md — working context for synapse

This file orients a Claude Code session on **synapse**. Read it before making
changes. The full decision history is in `CHANGELOG.md`; deep docs in `docs/`.

## What this is

synapse is a composable, **async-first, provider-neutral agent framework**:
build an agent once, call it anywhere — in-process, streaming, or over the
**A2A protocol** (v0.3.0). Status: **experimental (v0.7)**, ~6k LOC, 151 offline
tests. It has broad mechanism coverage; the honest weak spots are written down
(see "Honest boundaries" below and README → *Status & honest limitations*).

## How to work here (quality gate — keep all three green)

```bash
pip install -e ".[dev,anthropic,openai]"
ruff check src tests examples
mypy                 # synapse ships py.typed — type-checking must stay clean
pytest               # fully offline (fakes); no API key needed
pytest -m integration   # gated; only runs with ANTHROPIC_API_KEY / OPENAI_API_KEY
```

- Core is **dependency-free**; backends/extras are optional (`anthropic`,
  `openai`, `server`, `mcp`, `otel`).
- Tests are offline and deterministic via `ScriptedModel` / `EchoModel`. Every
  new capability gets a test, including its **failure** mode.
- `RunContext` is the **canonical** run config; `Agent.run`/`astream` are thin
  forwarders over `Agent.arun`. Add new run options to `RunContext`, not a
  parallel surface.
- Backends implement `async def generate` (+ optional `stream`) on `Model`.
  Tools may be sync or async; the loop offloads sync work.

## Architecture (src/synapse/)

- `agent.py` — `Agent` (the spine); `runtime.py` — the async loop + `RunContext`
  + `RunResult`, streaming is the primitive.
- `messages.py` (provider-neutral blocks: Text/Image/Document/ToolUse/ToolResult),
  `streaming.py`, `tool.py`, `models/` (base/anthropic/openai/scripted/resilient).
- Capabilities: `observability.py` (hooks/Usage), `tracing.py`, `memory.py`
  (+ `MemoryNamespace` for per-tenant isolation), `context.py` (Compactor),
  `results.py` (offload), `plan.py`, `journal.py` (idempotency),
  `recording.py` (run records), `verifiers.py`, `guardrails.py`, `human.py`
  (escalation), `skill.py`, `structured.py`, `evaluation.py`,
  `harness.py` + `selfharness.py`, `sandbox.py`, `router.py`, `team.py`,
  `registry.py`, `checkpoint.py`.
- `a2a/` — A2A v0.3.0 (spec/jsonrpc/dispatcher/asgi compliant server, client,
  push, task store) + a synapse-native `/run` convenience layer.

## Design & technical philosophy (the spine)

1. **Depth over breadth.** The framework already has wide mechanism coverage.
   The standing risk is "breadth outrunning a point of view." Prefer
   consolidation, a golden path, and saying *no* over adding another peripheral
   feature. Distinguish a real **first-principles** gap (context, planning,
   verification, idempotency, evidence) from peripheral breadth (one more
   provider/channel) — do the former, resist the latter.
2. **Honest naming.** A name/claim must match its substance. Where something is
   a first cut or has limits, say so in the docstring, README, and CHANGELOG —
   don't hide it behind optimistic naming. (e.g. the sandbox is process
   isolation, NOT a security boundary; the journal is identical-sequence
   idempotency, NOT distributed durable execution; keyword memory is not
   semantic.)
3. **Structure over convention for safety.** Isolation (per-tenant memory,
   sessions), guards, and boundaries are enforced structurally, not by "trying
   to be careful." Scope keys come from authn, never user-controllable input.
4. **Context is the scarce resource.** Don't dump large results into context;
   offload and fetch on demand. Treat "the agent gets dumber over a long run"
   as an engineering problem, not a complaint.

## Standing working preferences (from the build sessions)

- For each substantive change, end with a short **哲学反思 (philosophical
  reflection)** tying the change to first principles / design philosophy.
- Confirm scope before large or ambiguous edits; recommend, don't survey.
- Keep `Synapse` as the name (the maintainer chose it deliberately).
- After pushing, ensure a draft PR exists; keep the quality gate green.

## Honest boundaries / roadmap (not yet done — don't overclaim)

- Never validated end-to-end against a live model (integration tests are gated).
- No opinionated **default harness** content (the `Harness` surface exists, the
  curated default does not) — this is the biggest "give it a spine" gap.
- Compaction (naive prefix summary) and tool-search/router (keyword) are
  first-cut heuristics.
- No full durable execution (atomic mid-batch recovery), no global rate
  limiting/quotas, no secrets manager, no PyPI release.
- `run_sync` spins a worker-thread loop when called inside a running loop —
  fine for scripts, not for embedding in an async server.
