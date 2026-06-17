"""Tests for session + memory isolation (multi-tenant)."""

from __future__ import annotations

from synapse import (
    Agent,
    FileNamespace,
    HashingEmbedder,
    InMemoryNamespace,
    ScriptedModel,
    Session,
    VectorNamespace,
)
from synapse.messages import ToolResultBlock


# -- memory isolation --------------------------------------------------------


async def test_in_memory_namespace_isolates_scopes():
    ns = InMemoryNamespace()
    await ns.scope("tenant-a").add("A's secret is alpha")
    await ns.scope("tenant-b").add("B's secret is beta")

    assert await ns.scope("tenant-a").search("secret") == ["A's secret is alpha"]
    assert await ns.scope("tenant-b").search("secret") == ["B's secret is beta"]
    # a never sees b
    assert "beta" not in " ".join(await ns.scope("tenant-a").all())


async def test_same_key_returns_same_store():
    ns = InMemoryNamespace()
    await ns.scope("u1").add("remembered")
    assert await ns.scope("u1").all() == ["remembered"]  # stable across lookups


async def test_file_namespace_isolates_and_sanitizes(tmp_path):
    ns = FileNamespace(tmp_path)
    await ns.scope("../etc/passwd").add("escaped?")  # path traversal in the key
    await ns.scope("tenant-a").add("a-fact")
    # the malicious key is sanitized to a flat filename inside the dir
    files = [p.name for p in tmp_path.iterdir()]
    assert all(".." not in f and "/" not in f for f in files)
    assert await ns.scope("tenant-a").all() == ["a-fact"]


async def test_vector_namespace_isolates():
    ns = VectorNamespace(HashingEmbedder())
    await ns.scope("a").add("python async tips")
    await ns.scope("b").add("sourdough recipe")
    assert await ns.scope("a").search("async python") == ["python async tips"]
    assert await ns.scope("b").search("async python") == []  # b has nothing relevant


async def test_per_user_agent_recall_is_isolated():
    ns = InMemoryNamespace()
    await ns.scope("alice").add("Alice's project is payments")

    # Bob's agent is built on Bob's memory scope — cannot recall Alice's facts.
    bob = Agent(
        "asst",
        model=ScriptedModel([[("recall", {"query": "project"})], "done"]),
        memory=ns.scope("bob"),
    )
    result = await bob.arun("what's my project?")
    replies = [
        b.content
        for m in result.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert all("payments" not in c for c in replies if isinstance(c, str))


# -- session isolation -------------------------------------------------------


async def test_distinct_sessions_do_not_share_history():
    agent = Agent("a", model=ScriptedModel(["one", "two", "three", "four"]))
    alice = Session(scope="alice")
    bob = Session(scope="bob")

    await agent.arun("my name is Alice", session=alice)
    await agent.arun("my name is Bob", session=bob)

    alice_text = " ".join(m.text for m in alice.messages)
    bob_text = " ".join(m.text for m in bob.messages)
    assert "Alice" in alice_text and "Bob" not in alice_text
    assert "Bob" in bob_text and "Alice" not in bob_text
    assert alice.scope == "alice" and bob.scope == "bob"
