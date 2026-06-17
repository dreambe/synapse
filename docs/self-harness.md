# Self-Harness (experimental)

> Inspired by *Self-Harness: Harnesses That Improve Themselves* (Shanghai AI
> Lab). The idea is **change management for the harness**, not model
> self-evolution: the model/tools/budget/evaluator stay fixed, only the harness
> changes, and a change is kept only if it passes a regression gate.

## The harness as a versioned object

A `Harness` bundles the editable runtime surface around the model:

```python
from synapse import Harness
h = Harness(
    instructions="You are a precise assistant.",
    tool_descriptions={"search": "Search the web. Call before answering questions about current events."},
    max_iterations=20,
    max_repeated_tool_calls=3,      # loop guard (see Loop control)
    validate_tool_inputs=True,
)
agent = h.agent("assistant", model, tools=(search,))   # build an agent from it
```

Only **declared** surfaces are editable (`synapse.EDITABLE_SURFACES`):
`instructions`, `tool_descriptions`, `max_iterations`, the three loop guards,
and `validate_tool_inputs`. Permissions, billing, and connector auth are out of
bounds by construction. `h.with_edits({...})` returns a new, version-bumped
harness (and rejects edits to non-editable surfaces).

## The closed loop: mine → propose → promote

```python
from synapse import evolve, model_proposer, Case, contains

result = await evolve(
    harness=Harness(instructions="Answer questions."),
    model=model, tools=(search,),
    held_in=[Case("...", check=contains("...")), ...],
    held_out=[Case("...", check=contains("...")), ...],
    proposer=model_proposer(model),     # or any callable EvidencePack -> [HarnessEdit]
    rounds=3,
)
print(result.harness.version, result.promotions)
for rec in result.records:
    print("PROMOTED" if rec.promoted else "rejected", rec.edit.target_failure, "—", rec.reason)
```

1. **Weakness mining** — the suite is run; each failure becomes structured
   evidence (`mine_weaknesses`): a small action-oriented label set
   (`classify_failure`) built from `stop_reason` (`loop_detected`, `no_progress`,
   `tool_errors_exhausted`, `max_iterations`, …) plus `check_failed` / `no_output`,
   not raw logs.
2. **Proposal** — a proposer returns bounded `HarnessEdit`s, each with an audit
   record: `target_failure`, `edited_surface`, `expected_effect`,
   `regression_risk`, and `changes`. `model_proposer` asks an LLM for these as
   validated JSON (structured outputs); you can also pass a plain function.
3. **Promotion** — each candidate harness is re-evaluated on **held-in** and
   **held-out**. It is promoted only if **at least one split improves and
   neither regresses**; otherwise it's recorded and the active harness is
   unchanged. This is the gate that stops "locally better, globally worse".

## Honest boundaries

This is the *mechanism*, bounded like the paper: a fixed evaluator with a
**pass-rate non-regression** gate over a finite editable surface. It is not a
production-open self-improver — higher-risk surfaces need a stronger acceptance
gate than pass-rate, and stacking edits over a fixed benchmark can make a
harness brittle on open task streams. Reliable evidence is the precondition:
trajectories (`RunResult.messages`, `TracingHooks`), `stop_reason`, and a fact
source (`Checkpointer`, A2A `TaskStore`) are what make weakness mining possible.

See [`examples/self_harness.py`](../examples/self_harness.py).
