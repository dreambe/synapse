# Contributing to synapse

Thanks for your interest. synapse is pre-1.0 and experimental — the bar for
changes is *taste and honesty* as much as correctness.

## Principles

1. **Depth over breadth.** A new capability earns its place by being done well,
   not by ticking a pattern off a list. We would rather say "not yet" than ship
   a stub that pretends to be finished.
2. **Label maturity honestly.** If something is a first cut (a keyword
   heuristic, a best-effort mechanism), say so in the docstring and the README.
   No overclaiming in names, docs, or commit messages.
3. **Keep the core dependency-free.** New runtime dependencies belong behind an
   optional extra (`[anthropic]`, `[server]`, `[mcp]`), never in the core.
4. **One option surface.** New run options go on `RunContext`; don't grow a
   parallel set of keyword arguments.
5. **Async-first.** New backends implement `async def generate`; tools may be
   sync or async. Don't block the event loop — offload sync work.

## Development setup

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest          # the suite is fully offline — no API key required
ruff check src tests examples
```

## Tests

- Use `ScriptedModel` / `EchoModel` so tests stay offline and deterministic.
- Every new capability needs a test, including a test of its *failure* mode
  (timeout, denial, guardrail violation, etc.).
- CI runs `ruff` + `pytest` on Python 3.10–3.12; keep it green.

## Pull requests

Keep PRs focused. In the description, state what's solid vs first cut, and call
out anything you deliberately left undone.
