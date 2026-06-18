"""Cross-run memory: let an agent persist and recall facts beyond one session.

``Session`` only holds the current conversation. A :class:`Memory` survives
across runs and processes. Attach one to an agent and it gains ``remember`` and
``recall`` tools automatically, so the model drives its own memory the way
frontier file-/store-based memory works.
"""

from __future__ import annotations

import hashlib
import json
import math
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Process-local memory with keyword search. The zero-dependency default.

    Experimental / first cut: ``search`` is lexical overlap, not semantic.
    Implement :class:`Memory` over an embedding store for production recall.
    """

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


class Embedder(ABC):
    """Turns text into vectors. Implement over any embedding model."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashingEmbedder(Embedder):
    """A dependency-free hashing bag-of-words embedder.

    Deterministic and offline — token-overlap in vector space, a step up from
    raw substring search and a stand-in for tests/defaults. For genuine
    semantic recall use :class:`OpenAIEmbedder` (or your own ``Embedder``).
    """

    def __init__(self, dim: int = 256) -> None:
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for token in text.lower().split():
            h = int(hashlib.md5(token.encode()).hexdigest(), 16)  # noqa: S324 - not security
            v[h % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v))
        return [x / norm for x in v] if norm else v

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class OpenAIEmbedder(Embedder):
    """Embeddings via an OpenAI-compatible endpoint (optional ``[openai]``)."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        client: Any | None = None,
    ) -> None:
        self.model = model
        self._client = client
        self._base_url = base_url
        self._api_key = api_key

    def _get_client(self) -> Any:
        if self._client is None:
            import openai

            kwargs: dict[str, Any] = {}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            if self._api_key:
                kwargs["api_key"] = self._api_key
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def embed(self, texts: list[str]) -> list[list[float]]:
        resp = await self._get_client().embeddings.create(model=self.model, input=texts)
        return [d.embedding for d in resp.data]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


@dataclass
class _VectorItem:
    text: str
    vector: list[float]
    meta: dict = field(default_factory=dict)


class VectorMemory(Memory):
    """Semantic memory backed by a pluggable :class:`Embedder` (cosine search)."""

    def __init__(self, embedder: Embedder, *, min_score: float = 0.0) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self._items: list[_VectorItem] = []

    async def add(self, text: str, **meta: object) -> None:
        vector = (await self.embedder.embed([text]))[0]
        self._items.append(_VectorItem(text=text, vector=vector, meta=dict(meta)))

    async def search(self, query: str, k: int = 5) -> list[str]:
        if not self._items:
            return []
        qv = (await self.embedder.embed([query]))[0]
        scored = sorted(self._items, key=lambda it: _cosine(qv, it.vector), reverse=True)
        return [it.text for it in scored[:k] if _cosine(qv, it.vector) > self.min_score]

    async def all(self) -> list[str]:
        return [it.text for it in self._items]


def _safe_key(key: str) -> str:
    safe = "".join(c for c in key if c.isalnum() or c in ("-", "_", "."))
    while ".." in safe:
        safe = safe.replace("..", ".")
    safe = safe.strip(".")
    return safe[:128] or "default"


class MemoryNamespace(ABC):
    """Hands out an isolated :class:`Memory` per scope (tenant / user / session).

    The same key always returns the same store; different keys are disjoint —
    so one user's ``recall`` never sees another's facts. Use this whenever one
    agent serves multiple tenants or users.
    """

    @abstractmethod
    def scope(self, key: str) -> Memory: ...


class InMemoryNamespace(MemoryNamespace):
    """Process-local, scope-partitioned memory."""

    def __init__(self) -> None:
        self._scopes: dict[str, InMemoryMemory] = {}

    def scope(self, key: str) -> Memory:
        return self._scopes.setdefault(key, InMemoryMemory())


class FileNamespace(MemoryNamespace):
    """One JSONL file per scope under ``directory`` (survives restarts)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._scopes: dict[str, FileMemory] = {}

    def scope(self, key: str) -> Memory:
        if key not in self._scopes:
            self._scopes[key] = FileMemory(self.directory / f"{_safe_key(key)}.jsonl")
        return self._scopes[key]


class VectorNamespace(MemoryNamespace):
    """Scope-partitioned semantic memory sharing one embedder."""

    def __init__(self, embedder: "Embedder", *, min_score: float = 0.0) -> None:
        self.embedder = embedder
        self.min_score = min_score
        self._scopes: dict[str, VectorMemory] = {}

    def scope(self, key: str) -> Memory:
        if key not in self._scopes:
            self._scopes[key] = VectorMemory(self.embedder, min_score=self.min_score)
        return self._scopes[key]


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
