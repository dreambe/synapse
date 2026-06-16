"""Tests that the A2A layer strictly matches the protocol (v0.3.0)."""

from __future__ import annotations

import threading
import time

import pytest

from synapse import Agent, ScriptedModel
from synapse.a2a import A2AClient, A2ADispatcher, create_a2a_app
from synapse.a2a.spec import AgentCard, Message, TaskState


def _agent(script=None):
    return Agent("calc", description="adds", model=ScriptedModel(script or ["the answer is 4"]))


# -- serialization round-trips ----------------------------------------------


def test_message_roundtrip():
    msg = Message.user_text("hello")
    d = msg.to_dict()
    assert d["kind"] == "message" and d["role"] == "user"
    assert d["parts"][0] == {"kind": "text", "text": "hello"}
    back = Message.from_dict(d)
    assert back.text == "hello" and back.role == "user"


def test_agent_card_schema():
    card = AgentCard.from_dict(A2ADispatcher(_agent()).agent_card("http://x").to_dict())
    assert card.protocol_version == "0.3.0"
    assert card.capabilities.streaming is True
    assert card.url == "http://x"


# -- dispatcher: unary -------------------------------------------------------


async def test_message_send_returns_completed_task():
    disp = A2ADispatcher(_agent(["computed: 4"]))
    resp = await disp.handle(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {"message": Message.user_text("2+2?").to_dict()},
        }
    )
    result = resp["result"]
    assert resp["jsonrpc"] == "2.0" and resp["id"] == 1
    assert result["kind"] == "task"
    assert result["status"]["state"] == TaskState.COMPLETED
    assert result["status"]["message"]["role"] == "agent"
    assert result["status"]["message"]["parts"][0]["text"] == "computed: 4"
    assert result["artifacts"][0]["parts"][0]["text"] == "computed: 4"


async def test_unknown_method_is_method_not_found():
    disp = A2ADispatcher(_agent())
    resp = await disp.handle({"jsonrpc": "2.0", "id": 9, "method": "bogus", "params": {}})
    assert resp["error"]["code"] == -32601


async def test_tasks_get_and_not_found():
    disp = A2ADispatcher(_agent(["done"]))
    send = await disp.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "message/send",
         "params": {"message": Message.user_text("hi").to_dict()}}
    )
    task_id = send["result"]["id"]
    got = await disp.handle({"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task_id}})
    assert got["result"]["id"] == task_id

    missing = await disp.handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tasks/get", "params": {"id": "nope"}}
    )
    assert missing["error"]["code"] == -32001  # TaskNotFound


async def test_cancel_completed_task_is_not_cancelable():
    disp = A2ADispatcher(_agent(["done"]))
    send = await disp.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "message/send",
         "params": {"message": Message.user_text("hi").to_dict()}}
    )
    task_id = send["result"]["id"]
    resp = await disp.handle({"jsonrpc": "2.0", "id": 2, "method": "tasks/cancel", "params": {"id": task_id}})
    assert resp["error"]["code"] == -32002  # TaskNotCancelable


# -- dispatcher: streaming ---------------------------------------------------


async def test_message_stream_event_sequence():
    disp = A2ADispatcher(_agent(["streamed answer"]))
    envelopes = [
        env
        async for env in disp.stream(
            {"jsonrpc": "2.0", "id": 1, "method": "message/stream",
             "params": {"message": Message.user_text("go").to_dict()}}
        )
    ]
    results = [e["result"] for e in envelopes]
    kinds = [r["kind"] for r in results]
    assert kinds[0] == "task"  # initial submitted Task
    assert results[0]["status"]["state"] == TaskState.SUBMITTED
    assert "status-update" in kinds and "artifact-update" in kinds
    # last event is the terminal status-update
    assert results[-1]["kind"] == "status-update"
    assert results[-1]["final"] is True
    assert results[-1]["status"]["state"] == TaskState.COMPLETED
    assert results[-1]["status"]["message"]["parts"][0]["text"] == "streamed answer"


# -- end-to-end over HTTP (uvicorn) -----------------------------------------


@pytest.fixture
def a2a_server():
    uvicorn = pytest.importorskip("uvicorn")
    agent = Agent("remotecalc", description="adds numbers", model=ScriptedModel(["sum is 4"] * 50))
    config = uvicorn.Config(create_a2a_app(agent), host="127.0.0.1", port=8231, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    for _ in range(50):
        if server.started:
            break
        time.sleep(0.05)
    try:
        yield "http://127.0.0.1:8231"
    finally:
        server.should_exit = True
        thread.join(timeout=5)


def test_a2a_client_card_and_ask(a2a_server):
    client = A2AClient(a2a_server)
    card = client.card()
    assert card.name == "remotecalc"
    assert card.protocol_version == "0.3.0"
    assert client.ask("what is 2+2?") == "sum is 4"


async def test_a2a_client_stream(a2a_server):
    client = A2AClient(a2a_server)
    results = [r async for r in client.astream_message("go")]
    assert results[0]["kind"] == "task"
    assert results[-1]["kind"] == "status-update" and results[-1]["final"] is True


def test_a2a_client_as_tool_cross_ecosystem(a2a_server):
    # A synapse agent delegates to a standards-compliant A2A agent.
    remote = A2AClient(a2a_server)
    coordinator = Agent(
        "coord",
        model=ScriptedModel([[("ask_remotecalc", {"input": "add 2 and 2"})], "delegated"]),
        tools=[remote.as_tool()],
    )
    result = coordinator.run("please add")
    assert result.output == "delegated"
