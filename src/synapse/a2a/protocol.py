"""The agent-to-agent (A2A) wire protocol.

Two small JSON shapes plus an agent card. The card is published at
``/.well-known/agent.json`` so a peer can discover an agent's identity and
skills before talking to it; the run envelope carries a task in and the result
out.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class AgentCard:
    """A discovery document describing a remote agent."""

    name: str
    description: str = ""
    version: str = "0.1.0"
    skills: list[str] = field(default_factory=list)
    url: str | None = None
    protocol: str = "synapse-a2a/0.1"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AgentCard":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class RunRequest:
    """A task sent to a remote agent."""

    input: str
    session_id: str | None = None
    max_iterations: int = 12

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunRequest":
        return cls(
            input=data["input"],
            session_id=data.get("session_id"),
            max_iterations=data.get("max_iterations", 12),
        )


@dataclass
class RunResponse:
    """A remote agent's reply to a :class:`RunRequest`."""

    output: str
    agent: str
    iterations: int
    stop_reason: str
    session_id: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RunResponse":
        return cls(
            output=data["output"],
            agent=data["agent"],
            iterations=data.get("iterations", 0),
            stop_reason=data.get("stop_reason", "end_turn"),
            session_id=data.get("session_id"),
        )
