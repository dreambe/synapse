"""Pluggable persistence for A2A tasks.

The dispatcher keeps tasks here so they survive beyond a single request — and,
with :class:`FileTaskStore`, beyond a process restart. Tasks are stored as
plain dicts (the A2A ``Task`` wire form).
"""

from __future__ import annotations

import json
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


class TaskStore(ABC):
    @abstractmethod
    async def get(self, task_id: str) -> Optional[dict]: ...

    @abstractmethod
    async def put(self, task_id: str, task: dict) -> None: ...


class InMemoryTaskStore(TaskStore):
    """Process-local task store (default). Lost on restart."""

    def __init__(self) -> None:
        self._tasks: dict[str, dict] = {}

    async def get(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    async def put(self, task_id: str, task: dict) -> None:
        self._tasks[task_id] = task


class FileTaskStore(TaskStore):
    """Persist each task as a JSON file under ``directory`` (survives restarts)."""

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path(self, task_id: str) -> Path:
        safe = "".join(c for c in task_id if c.isalnum() or c in ("-", "_")) or "task"
        return self.directory / f"{safe}.json"

    async def get(self, task_id: str) -> Optional[dict]:
        path = self._path(task_id)
        if not path.exists():
            return None
        return json.loads(path.read_text())

    async def put(self, task_id: str, task: dict) -> None:
        with self._lock:
            self._path(task_id).write_text(json.dumps(task))
