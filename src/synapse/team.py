"""Teams: a coordinator delegating to peer agents with shared state.

First cut of multi-agent collaboration beyond simple delegation. A
:class:`Team` wires each member onto the coordinator as a delegation tool
(parallel fan-out comes for free from the async loop) and gives every agent a
shared :class:`Blackboard` so peers can communicate indirectly — post findings,
read what others posted — instead of only returning to the coordinator.
"""

from __future__ import annotations

import asyncio

from .agent import Agent
from .runtime import RunResult
from .tool import Tool


class Blackboard:
    """A shared, async-safe key/value space for inter-agent communication."""

    def __init__(self) -> None:
        self._data: dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def write(self, key: str, value: str) -> None:
        async with self._lock:
            self._data[key] = value

    async def read(self, key: str) -> str | None:
        async with self._lock:
            return self._data.get(key)

    async def dump(self) -> dict[str, str]:
        async with self._lock:
            return dict(self._data)

    def tools(self) -> list[Tool]:
        """Tools that let an agent post to and read from the blackboard."""

        async def post_note(key: str, value: str) -> str:
            await self.write(key, value)
            return f"posted note {key!r}"

        async def read_notes(key: str) -> str:
            value = await self.read(key)
            return value if value is not None else f"(no note for {key!r})"

        return [
            Tool(
                name="post_note",
                description="Share a finding with the team by posting it under a key.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "Note name."},
                        "value": {"type": "string", "description": "Note contents."},
                    },
                    "required": ["key", "value"],
                },
                func=post_note,
            ),
            Tool(
                name="read_notes",
                description="Read a teammate's shared note by key.",
                parameters={
                    "type": "object",
                    "properties": {"key": {"type": "string", "description": "Note name."}},
                    "required": ["key"],
                },
                func=read_notes,
            ),
        ]


class Team:
    """A coordinator plus member agents that collaborate via a blackboard."""

    def __init__(
        self,
        coordinator: Agent,
        members: list[Agent],
        *,
        shared_blackboard: bool = True,
    ) -> None:
        self.coordinator = coordinator
        self.members = members
        self.blackboard = Blackboard() if shared_blackboard else None

        if self.blackboard is not None:
            bb_tools = self.blackboard.tools()
            for agent in [coordinator, *members]:
                for t in bb_tools:
                    if t.name not in agent.tool_map:
                        agent.add_tool(t)

        existing = coordinator.tool_map
        for member in members:
            delegate = member.as_tool()
            if delegate.name not in existing:
                coordinator.add_tool(delegate)

    async def arun(self, task: str, **kwargs) -> RunResult:
        return await self.coordinator.arun(task, **kwargs)

    def run(self, task: str, **kwargs) -> RunResult:
        return self.coordinator.run(task, **kwargs)
