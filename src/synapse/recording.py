"""Persisted, queryable run records — the agent's "fact source".

Self-Harness and any failure analysis need reliable evidence: what was asked,
what happened, why it stopped. :class:`RunRecorder` is a hook that writes a
:class:`RunRecord` (input, output, trajectory, usage, stop reason) to a
:class:`RunStore` at the end of every run, unifying the trajectory into one
queryable place. Race-free: it derives everything from the final ``RunResult``.
"""

from __future__ import annotations

import json
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .observability import Hooks, Usage


@dataclass
class RunRecord:
    run_id: str
    agent: str
    input: str
    output: str
    stop_reason: str
    iterations: int
    input_tokens: int
    output_tokens: int
    messages: list[dict] = field(default_factory=list)
    ended_at: str = ""

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "agent": self.agent,
            "input": self.input,
            "output": self.output,
            "stop_reason": self.stop_reason,
            "iterations": self.iterations,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "messages": self.messages,
            "ended_at": self.ended_at,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunRecord":
        return cls(**d)


class RunStore(ABC):
    @abstractmethod
    async def put(self, record: RunRecord) -> None: ...

    @abstractmethod
    async def get(self, run_id: str) -> Optional[RunRecord]: ...

    @abstractmethod
    async def list(self) -> list[RunRecord]: ...


class InMemoryRunStore(RunStore):
    def __init__(self) -> None:
        self._records: dict[str, RunRecord] = {}

    async def put(self, record: RunRecord) -> None:
        self._records[record.run_id] = record

    async def get(self, run_id: str) -> Optional[RunRecord]:
        return self._records.get(run_id)

    async def list(self) -> list[RunRecord]:
        return list(self._records.values())


class FileRunStore(RunStore):
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    async def put(self, record: RunRecord) -> None:
        with self._lock:
            (self.directory / f"{record.run_id}.json").write_text(json.dumps(record.to_dict()))

    async def get(self, run_id: str) -> Optional[RunRecord]:
        path = self.directory / f"{run_id}.json"
        return RunRecord.from_dict(json.loads(path.read_text())) if path.exists() else None

    async def list(self) -> list[RunRecord]:
        return [
            RunRecord.from_dict(json.loads(p.read_text()))
            for p in sorted(self.directory.glob("*.json"))
        ]


class RunRecorder(Hooks):
    """A hook that records each completed run into a :class:`RunStore`."""

    def __init__(self, store: RunStore) -> None:
        self.store = store

    async def on_run_end(self, result) -> None:  # RunResult
        first_user = next((m.text for m in result.messages if m.role == "user"), "")
        usage: Usage = result.usage
        record = RunRecord(
            run_id="run_" + uuid.uuid4().hex[:12],
            agent=result.agent,
            input=first_user,
            output=result.output,
            stop_reason=result.stop_reason,
            iterations=result.iterations,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            messages=[m.to_dict() for m in result.messages],
            ended_at=datetime.now(timezone.utc).isoformat(),
        )
        await self.store.put(record)
