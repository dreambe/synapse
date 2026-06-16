"""Cross-run memory: let an agent persist and recall facts beyond one session.

``Session`` only holds the current conversation. A :class:`Memory` survives
across runs and processes. Attach one to an agent and it gains ``remember`` and
``recall`` tools automatically, so the model drives its own memory the way
frontier file-/store-based memory works.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .tool import Tool


@dataclass
class MemoryItem:
    text: str
    meta: dict


class Memory(ABC):
    """Pluggable long-term memory backend."""

    @abstractmethod
    async def add(self, text: str, **meta: object) -> None: ...

    @abstractmethod
    async def search(self, query: str, k: int = 5) -> list[str]: ...

    @abstractmethod
    async def all(self) -> list[str]: ...


def _score(query: str, text: str) -> int:
    """A tiny lexical overlap score — enough for a default with no deps."""
    q = {w for w in query.lower().split() if w}
    t = text.lower()
    return sum(1 for w in q if w in t)


class InMemoryMemory(Memory):
    """Process-local memory with keyword search. The zero-dependency default."""

    def __init__(self) -> None:
        self._items: list[MemoryItem] = []

    async def add(self, text: str, **meta: object) -> None:
        self._items.append(MemoryItem(text=text, meta=dict(meta)))

    async def search(self, query: str, k: int = 5) -> list[str]:
        ranked = sorted(self._items, key=lambda it: _score(query, it.text), reverse=True)
        return [it.text for it in ranked[:k] if _score(query, it.text) > 0]

    async def all(self) -> list[str]:
        return [it.text for it in self._items]


class FileMemory(Memory):
    """Memory persisted to a JSONL file — survives process restarts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("")

    def _read(self) -> list[MemoryItem]:
        items: list[MemoryItem] = []
        for line in self.path.read_text().splitlines():
            if line.strip():
                data = json.loads(line)
                items.append(MemoryItem(text=data["text"], meta=data.get("meta", {})))
        return items

    async def add(self, text: str, **meta: object) -> None:
        with self._lock:
            with self.path.open("a") as fh:
                fh.write(json.dumps({"text": text, "meta": meta}) + "\n")

    async def search(self, query: str, k: int = 5) -> list[str]:
        items = self._read()
        ranked = sorted(items, key=lambda it: _score(query, it.text), reverse=True)
        return [it.text for it in ranked[:k] if _score(query, it.text) > 0]

    async def all(self) -> list[str]:
        return [it.text for it in self._read()]


def memory_tools(memory: Memory) -> list[Tool]:
    """Build ``remember``/``recall`` tools backed by ``memory``."""

    async def remember(fact: str) -> str:
        await memory.add(fact)
        return "ok, remembered"

    async def recall(query: str) -> str:
        hits = await memory.search(query)
        return "\n".join(f"- {h}" for h in hits) if hits else "(nothing relevant remembered)"

    return [
        Tool(
            name="remember",
            description="Store a durable fact for later recall across sessions.",
            parameters={
                "type": "object",
                "properties": {"fact": {"type": "string", "description": "The fact to store."}},
                "required": ["fact"],
            },
            func=remember,
        ),
        Tool(
            name="recall",
            description="Search durable memory for facts relevant to a query.",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "What to look up."}},
                "required": ["query"],
            },
            func=recall,
        ),
    ]
