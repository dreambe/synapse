"""Tests for the upgraded Compactor (token trigger, keep-first, structured)."""

from __future__ import annotations

import asyncio

from synapse import Compactor, ScriptedModel
from synapse.messages import Message


def _convo(n: int) -> list[Message]:
    msgs = [Message(role="user", content="ORIGINAL TASK: build the thing")]
    for i in range(n):
        role = "assistant" if i % 2 else "user"
        msgs.append(Message(role=role, content=f"turn {i} " + "x" * 50))
    return msgs


def test_token_trigger_fires_even_with_few_messages():
    # Few messages but huge — message-count trigger wouldn't fire; token one does.
    big = [
        Message(role="user", content="ORIGINAL TASK"),
        Message(role="assistant", content="y" * 8000),
        Message(role="user", content="a"),
        Message(role="assistant", content="b"),
        Message(role="user", content="c"),
        Message(role="assistant", content="d"),
        Message(role="user", content="e"),
        Message(role="assistant", content="f"),
        Message(role="user", content="g"),
        Message(role="assistant", content="h"),
        Message(role="user", content="i"),
    ]
    comp = Compactor(ScriptedModel(["RECAP"]), trigger_messages=999, trigger_tokens=500, keep_recent=3)
    out = asyncio.run(comp.maybe_compact(big))
    assert len(out) < len(big)
    assert "RECAP" in out[1].text  # recap sits right after the preserved first turn


def test_first_turn_is_preserved_verbatim():
    convo = _convo(40)
    comp = Compactor(ScriptedModel(["RECAP"]), trigger_messages=24, keep_recent=6)
    out = asyncio.run(comp.maybe_compact(convo))
    assert out[0].text.startswith("ORIGINAL TASK")
    assert "[Summary of earlier conversation]" in out[1].text
    assert len(out) == 1 + 1 + 6  # first + recap + kept tail


def test_no_compaction_under_threshold():
    convo = _convo(6)
    comp = Compactor(ScriptedModel(["RECAP"]), trigger_messages=24)
    out = asyncio.run(comp.maybe_compact(convo))
    assert out == convo  # untouched
