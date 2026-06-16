"""Checkpointing: persist run state so a run can survive a crash and resume.

A :class:`Checkpointer` saves the message history under a ``run_id`` after every
turn. Pass a checkpointer plus a ``run_id`` to a run and an interrupted run can
be picked up where it left off — the durable-execution foundation that
separates a prototype from a production agent.
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path

from .messages import Message, message_from_dict


class Checkpointer(ABC):
    """Persist and restore the message history of a run by id."""

    @abstractmethod
    async def save(self, run_id: str, messages: list[Message]) -> None: ...

    @abstractmethod
    async def load(self, run_id: str) -> list[Message] | None: ...


class InMemoryCheckpointer(Checkpointer):
    """Keeps checkpoints in a dict — useful for tests and single-process use."""

    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {}

    async def save(self, run_id: str, messages: list[Message]) -> None:
        self._store[run_id] = [m.to_dict() for m in messages]

    async def load(self, run_id: str) -> list[Message] | None:
        raw = self._store.get(run_id)
        if raw is None:
            return None
        return [message_from_dict(m) for m in raw]


class FileCheckpointer(Checkpointer):
    """Stores each run's history as a JSON file under ``directory``."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, run_id: str) -> Path:
        safe = "".join(c for c in run_id if c.isalnum() or c in ("-", "_")) or "run"
        return self.directory / f"{safe}.json"

    async def save(self, run_id: str, messages: list[Message]) -> None:
        payload = json.dumps([m.to_dict() for m in messages])
        with self._lock:
            self._path(run_id).write_text(payload)

    async def load(self, run_id: str) -> list[Message] | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        raw = json.loads(path.read_text())
        return [message_from_dict(m) for m in raw]
