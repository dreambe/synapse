"""Context offloading: keep large tool results out of the model's context.

Context is an agent's scarcest resource ("context rot" — quality degrades as it
fills). Instead of dumping a 50KB tool result verbatim into the transcript, the
run loop can offload anything over a threshold to a :class:`ResultStore`,
leaving a short preview + a reference in context, and expose a ``fetch_result``
tool so the model pulls the full (or a filtered slice of the) result on demand.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class ResultStore(ABC):
    @abstractmethod
    async def put(self, text: str) -> str:
        """Store ``text`` and return a reference id."""

    @abstractmethod
    async def get(self, ref: str) -> Optional[str]:
        """Retrieve text by reference id (None if unknown)."""


class InMemoryResultStore(ResultStore):
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    async def put(self, text: str) -> str:
        ref = "res_" + uuid.uuid4().hex[:12]
        self._data[ref] = text
        return ref

    async def get(self, ref: str) -> Optional[str]:
        return self._data.get(ref)


class FileResultStore(ResultStore):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    async def put(self, text: str) -> str:
        ref = "res_" + uuid.uuid4().hex[:12]
        (self.directory / f"{ref}.txt").write_text(text)
        return ref

    async def get(self, ref: str) -> Optional[str]:
        path = self.directory / f"{ref}.txt"
        return path.read_text() if path.exists() else None


def make_preview(text: str, ref: str, *, preview_chars: int = 280) -> str:
    """The in-context stand-in for an offloaded result."""
    head = text[:preview_chars]
    more = "\n…" if len(text) > preview_chars else ""
    return (
        f"[Large result offloaded as {ref} ({len(text)} chars). Preview:\n"
        f"{head}{more}\n"
        f"Call fetch_result('{ref}', contains='…') to read the full result.]"
    )
