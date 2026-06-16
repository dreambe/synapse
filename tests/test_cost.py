"""Tests for token→money cost accounting."""

from __future__ import annotations

from synapse import (
    Agent,
    Cost,
    CostTracker,
    ModelPrice,
    Usage,
    estimate_cost,
)
from synapse.cost import resolve_model_id
from synapse.messages import Message, TextBlock
from synapse.models.base import Model, ModelResponse
from synapse.models.resilient import RetryModel


class PricedModel(Model):
    """A fake backend that names a real model id and reports usage."""

    def __init__(self, model="claude-opus-4-8", *, in_tok=1000, out_tok=500) -> None:
        self.model = model
        self.in_tok = in_tok
        self.out_tok = out_tok

    async def generate(self, *, system, messages, tools):
        return ModelResponse(
            Message("assistant", [TextBlock("answer")]),
            usage=Usage(self.in_tok, self.out_tok),
        )


def test_estimate_known_model():
    cost = estimate_cost(Usage(1_000_000, 1_000_000), "claude-opus-4-8")
    assert cost.priced
    assert cost.input_cost == 5.0
    assert cost.output_cost == 25.0
    assert cost.total == 30.0
    assert cost.to_dict()["currency"] == "USD"


def test_estimate_unknown_model_is_unpriced():
    cost = estimate_cost(Usage(1000, 1000), "some-random-model")
    assert cost.priced is False
    assert cost.total == 0.0


def test_dated_model_id_matches_by_prefix():
    cost = estimate_cost(Usage(1_000_000, 0), "claude-haiku-4-5-20251001")
    assert cost.priced and cost.input_cost == 1.0


def test_custom_pricing_override():
    pricing = {"my-model": ModelPrice(2.0, 8.0)}
    cost = estimate_cost(Usage(1_000_000, 1_000_000), "my-model", pricing=pricing)
    assert cost.total == 10.0


def test_resolve_model_id_unwraps_retry():
    base = PricedModel("claude-sonnet-4-6")
    assert resolve_model_id(base) == "claude-sonnet-4-6"
    assert resolve_model_id(RetryModel(base)) == "claude-sonnet-4-6"


async def test_run_result_has_cost():
    agent = Agent("c", model=PricedModel("claude-opus-4-8", in_tok=1_000_000, out_tok=1_000_000))
    result = await agent.arun("hi")
    assert isinstance(result.cost, Cost)
    assert result.cost.priced and result.cost.total == 30.0


async def test_cost_tracker_accumulates():
    tracker = CostTracker()
    agent = Agent("c", model=PricedModel("claude-opus-4-8", in_tok=1_000_000, out_tok=0))
    await agent.arun("a", hooks=tracker)
    await agent.arun("b", hooks=tracker)
    assert tracker.runs == 2
    assert tracker.usage.input_tokens == 2_000_000
    assert tracker.total_cost == 10.0  # 2 × $5


async def test_a2a_task_reports_cost_metadata():
    from synapse.a2a import A2ADispatcher
    from synapse.a2a.spec import Message as A2AMessage

    agent = Agent("c", model=PricedModel("claude-opus-4-8", in_tok=1_000_000, out_tok=0))
    disp = A2ADispatcher(agent)
    resp = await disp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {"message": A2AMessage.user_text("hi").to_dict()},
        }
    )
    meta = resp["result"]["metadata"]
    assert meta["usage"]["inputTokens"] == 1_000_000
    assert meta["cost"]["total_cost"] == 5.0
    assert meta["cost"]["model"] == "claude-opus-4-8"
