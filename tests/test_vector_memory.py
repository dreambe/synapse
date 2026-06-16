"""Tests for vector (embedding-based) memory."""

from __future__ import annotations

from synapse import Agent, HashingEmbedder, ScriptedModel, VectorMemory
from synapse.memory import Embedder


class FakeEmbedder(Embedder):
    """Deterministic 3-d embedder keyed on topic words — for ranking tests."""

    async def embed(self, texts):
        out = []
        for t in texts:
            low = t.lower()
            out.append(
                [
                    1.0 if "cat" in low else 0.0,
                    1.0 if "finance" in low else 0.0,
                    1.0 if "weather" in low else 0.0,
                ]
            )
        return out


async def test_vector_memory_ranks_by_similarity():
    mem = VectorMemory(FakeEmbedder())
    await mem.add("cats are great pets")
    await mem.add("finance and markets")
    await mem.add("the weather today")
    hits = await mem.search("tell me about my cat", k=1)
    assert hits == ["cats are great pets"]


async def test_vector_memory_empty():
    mem = VectorMemory(FakeEmbedder())
    assert await mem.search("anything") == []


async def test_hashing_embedder_overlap():
    mem = VectorMemory(HashingEmbedder(dim=64))
    await mem.add("python async programming")
    await mem.add("baking sourdough bread")
    hits = await mem.search("async python tips", k=1)
    assert hits == ["python async programming"]


async def test_vector_memory_drives_recall_tool():
    mem = VectorMemory(HashingEmbedder())
    await mem.add("the launch code is alpha-7")
    agent = Agent(
        "m",
        model=ScriptedModel([[("recall", {"query": "launch code"})], "answered"]),
        memory=mem,
    )
    result = await agent.arun("what's the launch code?")
    assert result.output == "answered"
    tool_results = [
        b.content
        for msg in result.messages
        for b in msg.content
        if getattr(b, "type", None) == "tool_result"
    ]
    assert any("alpha-7" in c for c in tool_results if isinstance(c, str))
