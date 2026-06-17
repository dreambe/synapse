"""Idempotent tool execution: don't fire a side effect twice on replay.

Checkpoint/resume restores the transcript, but a crash *mid-turn* (after a tool
ran, before its result was saved) would re-execute that tool on resume — double
-sending an email, re-charging a card. An :class:`ExecutionJournal` records each
tool result under a stable key; on replay, a matching key returns the recorded
result instead of running the tool again.

Scope (honest): this makes replay of an *identical call sequence* idempotent —
keyed by run id + tool name + arguments + the call's ordinal within the run. It
is not full distributed durable execution; it's the safety that stops
side-effectful tools from double-firing.
"""

from __future__ import annotations

import hashlib
import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


def call_key(run_id: str, name: str, arguments: dict, ordinal: int) -> str:
    payload = json.dumps(
        {"run": run_id, "name": name, "args": arguments, "n": ordinal},
        sort_keys=True,
        default=str,
    )
    return "jk_" + hashlib.sha256(payload.encode()).hexdigest()[:24]


class ExecutionJournal(ABC):
    @abstractmethod
    async def lookup(self, key: str) -> Optional[dict]:
        """Return a recorded result dict for ``key`` (None if not recorded)."""

    @abstractmethod
    async def record(self, key: str, result: dict) -> None:
        """Persist a tool result under ``key``."""


class InMemoryJournal(ExecutionJournal):
    def __init__(self) -> None:
        self._data: dict[str, dict] = {}

    async def lookup(self, key: str) -> Optional[dict]:
        return self._data.get(key)

    async def record(self, key: str, result: dict) -> None:
        self._data[key] = result


class FileJournal(ExecutionJournal):
    """One JSON file per run; survives restarts."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _read(self) -> dict:
        return json.loads(self.path.read_text()) if self.path.exists() else {}

    async def lookup(self, key: str) -> Optional[dict]:
        return self._read().get(key)

    async def record(self, key: str, result: dict) -> None:
        with self._lock:
            data = self._read()
            data[key] = result
            self.path.write_text(json.dumps(data))
