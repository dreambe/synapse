"""Tests for pluggable A2A task persistence."""

from __future__ import annotations

from synapse import Agent, ScriptedModel
from synapse.a2a import A2ADispatcher, FileTaskStore, InMemoryTaskStore
from synapse.a2a.spec import Message


def _send(disp, text, rpc_id=1):
    return {
        "jsonrpc": "2.0",
        "id": rpc_id,
        "method": "message/send",
        "params": {"message": Message.user_text(text).to_dict()},
    }


async def test_in_memory_store_default():
    disp = A2ADispatcher(Agent("a", model=ScriptedModel(["done"])))
    send = await disp.handle(_send(disp, "hi"))
    task_id = send["result"]["id"]
    got = await disp.handle(
        {"jsonrpc": "2.0", "id": 2, "method": "tasks/get", "params": {"id": task_id}}
    )
    assert got["result"]["id"] == task_id
    assert got["result"]["status"]["state"] == "completed"


async def test_file_store_persists_across_dispatchers(tmp_path):
    store = FileTaskStore(tmp_path)
    agent = Agent("a", model=ScriptedModel(["persisted"] * 5))

    disp1 = A2ADispatcher(agent, task_store=store)
    send = await disp1.handle(_send(disp1, "hi"))
    task_id = send["result"]["id"]

    # A brand-new dispatcher backed by the same store can still fetch the task.
    disp2 = A2ADispatcher(agent, task_store=FileTaskStore(tmp_path))
    got = await disp2.handle(
        {"jsonrpc": "2.0", "id": 9, "method": "tasks/get", "params": {"id": task_id}}
    )
    assert got["result"]["id"] == task_id
    assert (tmp_path / f"{task_id}.json").exists()


async def test_cancel_updates_store():
    store = InMemoryTaskStore()
    disp = A2ADispatcher(Agent("a", model=ScriptedModel(["done"])), task_store=store)
    send = await disp.handle(_send(disp, "hi"))
    task_id = send["result"]["id"]
    # completed task → not cancelable
    resp = await disp.handle(
        {"jsonrpc": "2.0", "id": 3, "method": "tasks/cancel", "params": {"id": task_id}}
    )
    assert resp["error"]["code"] == -32002
