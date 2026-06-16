from synapse import Agent, ScriptedModel, tool
from synapse.a2a import AgentServer, RemoteAgent

import pytest


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@pytest.fixture
def served_agent():
    model = ScriptedModel([[("add", {"a": 2, "b": 2})], "result is 4"] * 50)
    agent = Agent("remote-calc", description="Adds numbers.", model=model, tools=[add])
    server = AgentServer(agent, port=0).start_background()
    try:
        yield server
    finally:
        server.stop()


def test_remote_card_discovery(served_agent):
    client = RemoteAgent(served_agent.url)
    card = client.card()
    assert card.name == "remote-calc"
    assert card.description == "Adds numbers."
    assert "add" in card.skills
    assert card.protocol == "synapse-a2a/0.1"


def test_remote_run(served_agent):
    client = RemoteAgent(served_agent.url)
    response = client.run("what is 2 + 2?")
    assert response.output == "result is 4"
    assert response.agent == "remote-calc"
    assert response.stop_reason == "end_turn"


async def test_remote_agent_as_tool(served_agent):
    # A local coordinator delegates to the remote agent via a tool.
    remote = RemoteAgent(served_agent.url)
    delegate = remote.as_tool()
    assert delegate.name == "ask_remote-calc"
    assert delegate.is_async
    assert await delegate.invoke(input="add 2 and 2") == "result is 4"


def test_in_process_a2a_as_tool():
    # No network: one agent calls another directly via as_tool().
    specialist = Agent(
        "adder",
        model=ScriptedModel(["the sum is 7"]),
        description="Adds numbers.",
    )
    coordinator = Agent(
        "boss",
        model=ScriptedModel([[("ask_adder", {"input": "add 3 and 4"})], "delegated; got it"]),
        tools=[specialist.as_tool()],
    )
    result = coordinator.run("please add 3 and 4")
    assert result.output == "delegated; got it"
