"""Agent-to-agent communication.

Two layers:

- **A2A-compliant** (``spec``, ``A2ADispatcher``, ``create_a2a_app``,
  ``A2AClient``): strictly implements the Agent2Agent protocol v0.3.0
  (JSON-RPC 2.0, agent card at ``/.well-known/agent-card.json``, task
  lifecycle, SSE streaming, push notifications). Interoperates with any
  A2A-compliant peer.
- **synapse-native** (``serve``, ``AgentServer``, ``RemoteAgent``,
  ``create_app``): a lightweight ``/run`` + ``/run/stream`` convenience layer.
"""

from __future__ import annotations

from .asgi import create_a2a_app, create_app
from .client import A2AClient, RemoteAgent, fetch_card
from .dispatcher import A2ADispatcher
from .protocol import AgentCard, RunRequest, RunResponse
from .server import AgentServer, serve
from .spec import (
    Artifact,
    Message,
    Task,
    TaskState,
    TaskStatus,
    TextPart,
)
from .spec import AgentCard as A2AAgentCard

__all__ = [
    # A2A-compliant
    "create_a2a_app",
    "A2ADispatcher",
    "A2AClient",
    "A2AAgentCard",
    "Message",
    "Task",
    "TaskState",
    "TaskStatus",
    "Artifact",
    "TextPart",
    # synapse-native
    "AgentCard",
    "RunRequest",
    "RunResponse",
    "AgentServer",
    "serve",
    "create_app",
    "RemoteAgent",
    "fetch_card",
]
