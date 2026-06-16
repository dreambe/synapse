"""Talk to a remote synapse agent over HTTP.

A :class:`RemoteAgent` is a local handle to an agent running in another
process. It can be called sync or async, or turned into a
:class:`~synapse.tool.Tool` so a *local* agent can delegate across the network
— the same ``as_tool`` pattern as in-process A2A, but over the wire. The async
path offloads the blocking HTTP call to a thread, so many remote calls can be
in flight at once.
"""

from __future__ import annotations

import asyncio
import json
import urllib.error
import urllib.request

from ..errors import A2AError
from ..tool import Tool
from .protocol import AgentCard, RunRequest, RunResponse
from .server import CARD_PATH


class RemoteAgent:
    """A client handle to an agent served by :func:`synapse.a2a.serve`."""

    def __init__(self, url: str, *, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._card: AgentCard | None = None

    # -- blocking HTTP primitives -------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.url + path,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise A2AError(f"request to {self.url}{path} failed: {exc}") from exc

    def _get(self, path: str) -> dict:
        try:
            with urllib.request.urlopen(self.url + path, timeout=self.timeout) as resp:
                return json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise A2AError(f"request to {self.url}{path} failed: {exc}") from exc

    # -- discovery -----------------------------------------------------------

    def card(self, *, refresh: bool = False) -> AgentCard:
        """Fetch and cache the remote agent's discovery card."""
        if self._card is None or refresh:
            self._card = AgentCard.from_dict(self._get(CARD_PATH))
        return self._card

    # -- invocation ----------------------------------------------------------

    def run(
        self,
        input: str,
        *,
        session_id: str | None = None,
        max_iterations: int = 12,
    ) -> RunResponse:
        """Run the remote agent on a task (blocking)."""
        request = RunRequest(
            input=input, session_id=session_id, max_iterations=max_iterations
        )
        return RunResponse.from_dict(self._post("/run", request.to_dict()))

    async def arun(
        self,
        input: str,
        *,
        session_id: str | None = None,
        max_iterations: int = 12,
    ) -> RunResponse:
        """Run the remote agent on a task without blocking the event loop."""
        return await asyncio.to_thread(
            self.run,
            input,
            session_id=session_id,
            max_iterations=max_iterations,
        )

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """Wrap the remote agent as a tool a local agent can call.

        The delegate is async, so a coordinator can fan out to several remote
        agents concurrently in a single turn.
        """
        card = self.card()
        remote = self
        tool_name = name or f"ask_{card.name}"

        async def _delegate(input: str) -> str:
            return (await remote.arun(input)).output

        _delegate.__name__ = tool_name
        return Tool(
            name=tool_name,
            description=description or card.description or f"Delegate to remote agent {card.name}.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The task or question to hand to the remote agent.",
                    }
                },
                "required": ["input"],
            },
            func=_delegate,
        )


def fetch_card(url: str, *, timeout: float = 60.0) -> AgentCard:
    """Fetch an agent card from a base URL without keeping a client."""
    return RemoteAgent(url, timeout=timeout).card()
