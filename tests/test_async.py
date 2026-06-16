import asyncio
import time

from synapse import Agent, ScriptedModel, tool
from synapse.models.base import Model, ModelResponse
from synapse.messages import Message, TextBlock


class SlowModel(Model):
    """A model that sleeps before answering — used to prove concurrency."""

    def __init__(self, answer: str, delay: float = 0.2) -> None:
        self.answer = answer
        self.delay = delay

    async def generate(self, *, system, messages, tools) -> ModelResponse:
        await asyncio.sleep(self.delay)
        return ModelResponse(Message("assistant", [TextBlock(self.answer)]))


async def test_arun_basic():
    agent = Agent("a", model=ScriptedModel(["hi there"]))
    result = await agent.arun("hello")
    assert result.output == "hi there"


async def test_concurrent_runs_do_not_serialize():
    # Three independent agents, each takes ~0.2s. Run concurrently → ~0.2s,
    # not ~0.6s. This is the inbound/outbound concurrency guarantee.
    agents = [Agent(f"a{i}", model=SlowModel(f"answer {i}", delay=0.2)) for i in range(3)]
    start = time.perf_counter()
    results = await asyncio.gather(*(a.arun("go") for a in agents))
    elapsed = time.perf_counter() - start
    assert [r.output for r in results] == ["answer 0", "answer 1", "answer 2"]
    assert elapsed < 0.5  # well under the 0.6s a serial run would take


async def test_parallel_tool_fan_out():
    # A coordinator delegates to two slow sub-agents *in one turn*. The two
    # delegations run concurrently, so the turn costs ~one delay, not two.
    slow_a = Agent("slow_a", model=SlowModel("from A", delay=0.2))
    slow_b = Agent("slow_b", model=SlowModel("from B", delay=0.2))

    coordinator = Agent(
        "coord",
        model=ScriptedModel(
            [
                [("ask_slow_a", {"input": "x"}), ("ask_slow_b", {"input": "y"})],
                "combined",
            ]
        ),
        tools=[slow_a.as_tool(), slow_b.as_tool()],
    )

    start = time.perf_counter()
    result = await coordinator.arun("delegate to both")
    elapsed = time.perf_counter() - start
    assert result.output == "combined"
    assert elapsed < 0.4  # both sub-agents ran in parallel, not back-to-back


async def test_async_tool_awaited():
    calls: list[str] = []

    @tool
    async def record(item: str) -> str:
        """Record an item asynchronously."""
        await asyncio.sleep(0)
        calls.append(item)
        return f"recorded {item}"

    agent = Agent(
        "rec",
        model=ScriptedModel([[("record", {"item": "widget"})], "done"]),
        tools=[record],
    )
    result = await agent.arun("record a widget")
    assert result.output == "done"
    assert calls == ["widget"]
