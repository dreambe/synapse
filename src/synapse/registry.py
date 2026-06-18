"""A registry for looking up agents by name — local or remote."""

from __future__ import annotations

from typing import Iterator

from .agent import Agent
from .errors import RegistryError


class AgentRegistry:
    """A named collection of agents.

    Useful for routing ("which agent handles this?"), for building multi-agent
    systems where agents discover each other, and for serving several agents
    behind one process.
    """

    def __init__(self) -> None:
        self._agents: dict[str, Agent] = {}

    def register(self, agent: Agent) -> Agent:
        if agent.name in self._agents:
            raise RegistryError(f"an agent named {agent.name!r} is already registered")
        self._agents[agent.name] = agent
        return agent

    def get(self, name: str) -> Agent:
        try:
            return self._agents[name]
        except KeyError as exc:
            raise RegistryError(f"no agent named {name!r}") from exc

    def names(self) -> list[str]:
        return list(self._agents)

    def __contains__(self, name: object) -> bool:
        return name in self._agents

    def __iter__(self) -> Iterator[Agent]:
        return iter(self._agents.values())

    def __len__(self) -> int:
        return len(self._agents)
