"""Tests for enterprise knowledge grounding + human escalation."""

from __future__ import annotations

from synapse import Agent, CallbackChannel, ScriptedModel, human_tool
from synapse.messages import Message, TextBlock, ToolResultBlock
from synapse.models.base import Model, ModelResponse


class CaptureModel(Model):
    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.systems: list[str] = []

    async def generate(self, *, system, messages, tools):
        self.systems.append(system or "")
        return ModelResponse(Message("assistant", [TextBlock(self.answer)]))


# -- grounding (preflight KG retrieval) -------------------------------------


async def test_grounding_injects_business_context():
    model = CaptureModel("planned")
    agent = Agent("dev", instructions="You write code.", model=model)

    def kg_lookup(requirement: str) -> str:
        # stand-in for a KG/MCP query keyed on the requirement
        assert "checkout" in requirement
        return "Repo: payments-svc. Backend only. Red line: never log PII."

    result = await agent.arun("add a checkout discount", grounding=kg_lookup)
    assert result.output == "planned"
    system = model.systems[0]
    assert "Business context" in system
    assert "payments-svc" in system and "never log PII" in system


async def test_grounding_supports_async_provider():
    model = CaptureModel()
    agent = Agent("dev", model=model)

    async def kg(requirement: str) -> str:
        return "Repo: web-app. Frontend + backend."

    await agent.arun("build a dashboard", grounding=kg)
    assert "web-app" in model.systems[0]


# -- human escalation -------------------------------------------------------


async def test_callback_channel_sync_and_async():
    sync_ch = CallbackChannel(lambda q, ctx: f"sync:{q}")
    assert await sync_ch.ask("hi") == "sync:hi"

    async def afn(q, ctx):
        return f"async:{q}"

    assert await CallbackChannel(afn).ask("hi") == "async:hi"


async def test_agent_escalates_via_ask_human():
    asked: list[str] = []

    def answer(question, context):
        asked.append(question)
        return "Use the payments-svc repo; do the backend."

    agent = Agent(
        "dev",
        model=ScriptedModel(
            [
                [("ask_human", {"question": "Which repo for the checkout change?"})],
                "Got it — implementing in payments-svc backend.",
            ]
        ),
        tools=[human_tool(CallbackChannel(answer))],
    )
    result = await agent.arun("change checkout")
    assert result.output.startswith("Got it")
    assert asked == ["Which repo for the checkout change?"]
    replies = [
        b.content
        for m in result.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert any("payments-svc" in c for c in replies if isinstance(c, str))


async def test_grounding_and_escalation_compose():
    # Grounding provides redlines; the agent escalates when unsure, both at once.
    model = ScriptedModel(
        [[("ask_human", {"question": "Is touching auth allowed here?"})], "proceeding within redlines"]
    )
    agent = Agent("dev", instructions="You write code.", model=model,
                  tools=[human_tool(CallbackChannel(lambda q, c: "Yes, but no schema changes."))])
    result = await agent.arun(
        "tweak login", grounding=lambda r: "Repo: auth-svc. Red line: no DB schema changes."
    )
    assert result.output == "proceeding within redlines"
