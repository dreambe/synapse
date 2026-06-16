"""Cost accounting: turn token usage into money.

Every run already tracks :class:`~synapse.observability.Usage` (input/output
tokens). This module prices that usage per model, attaches a :class:`Cost` to
each :class:`~synapse.runtime.RunResult`, and offers a :class:`CostTracker`
hook to sum spend across many runs.

Prices are USD per 1M tokens (Anthropic list pricing, cached — override per run
via ``RunContext.pricing`` for other providers, negotiated rates, or updates).
Unknown models are reported with ``priced=False`` and zero cost rather than a
wrong number — honesty over a confident guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .observability import Hooks, Usage


@dataclass(frozen=True)
class ModelPrice:
    """USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float


# Cached Anthropic list pricing (USD / 1M tokens).
PRICES: dict[str, ModelPrice] = {
    "claude-fable-5": ModelPrice(10.0, 50.0),
    "claude-opus-4-8": ModelPrice(5.0, 25.0),
    "claude-opus-4-7": ModelPrice(5.0, 25.0),
    "claude-opus-4-6": ModelPrice(5.0, 25.0),
    "claude-opus-4-5": ModelPrice(5.0, 25.0),
    "claude-sonnet-4-6": ModelPrice(3.0, 15.0),
    "claude-sonnet-4-5": ModelPrice(3.0, 15.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
}


@dataclass
class Cost:
    """The money a run cost, broken down by input/output."""

    model: str
    input_tokens: int
    output_tokens: int
    input_cost: float
    output_cost: float
    priced: bool = True

    @property
    def total(self) -> float:
        return round(self.input_cost + self.output_cost, 6)

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "input_cost": self.input_cost,
            "output_cost": self.output_cost,
            "total_cost": self.total,
            "currency": "USD",
            "priced": self.priced,
        }


def price_for(model: str, pricing: Optional[dict[str, ModelPrice]] = None) -> Optional[ModelPrice]:
    table = pricing or PRICES
    if model in table:
        return table[model]
    # tolerate dated/suffixed ids (e.g. claude-haiku-4-5-20251001)
    for key, price in table.items():
        if model.startswith(key):
            return price
    return None


def estimate_cost(
    usage: Usage,
    model: Optional[str],
    *,
    pricing: Optional[dict[str, ModelPrice]] = None,
) -> Cost:
    """Price ``usage`` for ``model``. Unknown model → ``priced=False``, zero."""
    price = price_for(model, pricing) if model else None
    if price is None:
        return Cost(model or "unknown", usage.input_tokens, usage.output_tokens, 0.0, 0.0, False)
    input_cost = round(usage.input_tokens / 1_000_000 * price.input_per_mtok, 6)
    output_cost = round(usage.output_tokens / 1_000_000 * price.output_per_mtok, 6)
    return Cost(model, usage.input_tokens, usage.output_tokens, input_cost, output_cost, True)


def resolve_model_id(model: object) -> Optional[str]:
    """Best-effort model id, unwrapping wrappers like ``RetryModel``."""
    inner = getattr(model, "model", None)
    if isinstance(inner, str):
        return inner
    if inner is not None and inner is not model:
        return resolve_model_id(inner)
    return None


class CostTracker(Hooks):
    """A hook that accumulates usage and cost across every run it observes."""

    def __init__(self) -> None:
        self.runs = 0
        self.usage = Usage()
        self.total_cost = 0.0

    async def on_run_end(self, result) -> None:  # noqa: ANN001 - RunResult (avoid cycle)
        self.runs += 1
        self.usage.add(result.usage)
        if getattr(result, "cost", None) is not None:
            self.total_cost = round(self.total_cost + result.cost.total, 6)
