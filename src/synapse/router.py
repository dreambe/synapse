"""Routing: pick the right agent for an input, then run it.

Two strategies, composable:

- **Rule-based** — register ``(predicate, agent)`` pairs; the first matching
  predicate wins.
- **Model-based** — hand the input to a small classifier model that picks an
  agent by name from the candidates.

A default agent handles anything unmatched.
"""

from __future__ import annotations

from typing import Callable

from .agent import Agent
from .errors import ConfigurationError
from .models import Model
from .runtime import RunResult

Predicate = Callable[[str], bool]


class Router:
    """Routes an input to one of several agents, then runs it."""

    def __init__(self, *, default: Agent | None = None) -> None:
        self._routes: list[tuple[Predicate, Agent]] = []
        self.default = default

    def add_route(self, predicate: Predicate, agent: Agent) -> "Router":
        self._routes.append((predicate, agent))
        return self

    def add_keyword_route(self, keywords: list[str], agent: Agent) -> "Router":
        lowered = [k.lower() for k in keywords]
        return self.add_route(lambda t: any(k in t.lower() for k in lowered), agent)

    def select(self, text: str) -> Agent:
        for predicate, agent in self._routes:
            if predicate(text):
                return agent
        if self.default is None:
            raise ConfigurationError("no route matched and no default agent set")
        return self.default

    async def arun(self, text: str, **kwargs) -> RunResult:
        return await self.select(text).arun(text, **kwargs)

    def run(self, text: str, **kwargs) -> RunResult:
        return self.select(text).run(text, **kwargs)


class ModelRouter:
    """Picks an agent by asking a model to choose among candidates by name."""

    def __init__(self, model: Model, agents: list[Agent], *, default: Agent | None = None) -> None:
        if not agents:
            raise ConfigurationError("ModelRouter needs at least one agent")
        self.model = model
        self.agents = {a.name: a for a in agents}
        self.default = default or agents[0]

    async def select(self, text: str) -> Agent:
        from .messages import Message

        roster = "\n".join(f"- {a.name}: {a.description}" for a in self.agents.values())
        system = (
            "You are a router. Reply with exactly one agent name from the list and "
            "nothing else.\n" + roster
        )
        response = await self.model.generate(
            system=system,
            messages=[Message(role="user", content=text)],
            tools=[],
        )
        choice = response.message.text.strip()
        return self.agents.get(choice, self.default)

    async def arun(self, text: str, **kwargs) -> RunResult:
        agent = await self.select(text)
        return await agent.arun(text, **kwargs)
