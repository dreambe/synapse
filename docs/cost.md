# Cost accounting

synapse prices the tokens a run consumes and attaches the result to every
`RunResult`, so you can see what each call (and each *being-called* A2A request)
cost — in money, not just tokens.

## How it works

1. Each model response reports `Usage(input_tokens, output_tokens)`.
2. The run loop sums usage across all turns into `RunResult.usage`.
3. At the end of the run it prices that usage for the agent's model, producing
   `RunResult.cost` (a `Cost`).

```python
result = await agent.arun("...")
print(result.usage.total_tokens)        # e.g. 12030
print(result.cost.total)                # e.g. 0.0421  (USD)
print(result.cost.to_dict())
# {'model': 'claude-opus-4-8', 'input_tokens': ..., 'output_tokens': ...,
#  'input_cost': ..., 'output_cost': ..., 'total_cost': 0.0421,
#  'currency': 'USD', 'priced': True}
```

If the model isn't in the price table, `cost.priced` is `False` and the totals
are `0.0` — synapse won't invent a number it can't stand behind.

## Pricing table

Built-in USD-per-1M-token list pricing lives in `synapse.cost.PRICES`
(Claude Opus/Sonnet/Haiku/Fable). Override per run for other providers,
negotiated rates, or price updates:

```python
from synapse import ModelPrice
pricing = {"my-llm": ModelPrice(input_per_mtok=2.0, output_per_mtok=8.0)}
agent.run("...", pricing=pricing)
```

`estimate_cost(usage, model, pricing=...)` is also exposed directly.

## Track spend across many runs

```python
from synapse import CostTracker
tracker = CostTracker()

for task in tasks:
    agent.run(task, hooks=tracker)

print(tracker.runs, tracker.usage.total_tokens, tracker.total_cost)
```

`CostTracker` is just a `Hooks` subclass — combine it with your own via
`CompositeHooks`.

## Cost of *being called* (A2A)

When a synapse agent is served over A2A, the completed `Task` carries usage and
cost in its `metadata`, so the **caller** sees what the work cost on the
**callee**:

```json
{
  "kind": "task",
  "status": { "state": "completed", ... },
  "metadata": {
    "usage": { "inputTokens": 1200, "outputTokens": 340, "totalTokens": 1540 },
    "cost":  { "model": "claude-opus-4-8", "total_cost": 0.0145, "currency": "USD", "priced": true }
  }
}
```

A client reads it straight off the task:

```python
from synapse.a2a import A2AClient
task = A2AClient(url).send_message("do the thing")
print(task["metadata"]["cost"]["total_cost"])
```

This gives you per-call attribution on both sides of an agent-to-agent
boundary — useful for chargeback, budgets, and spotting a runaway sub-agent.

> Caveat: cost is **list-price estimation** from token counts, not a billing
> source of truth. Prompt caching, batch discounts, and provider invoices can
> differ — reconcile against your provider for accounting.
