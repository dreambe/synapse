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
import threading
import urllib.error
import urllib.request
from typing import AsyncIterator

from ..errors import A2AError
from ..tool import Tool
from .protocol import AgentCard, RunRequest, RunResponse
from .server import CARD_PATH

_STREAM_DONE = object()


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

    async def astream(
        self,
        input: str,
        *,
        session_id: str | None = None,
        max_iterations: int = 12,
    ) -> AsyncIterator[dict]:
        """Stream a remote run as SSE, yielding parsed event dicts.

        The blocking HTTP read runs in a worker thread and feeds an
        ``asyncio.Queue`` so the consumer stays on the event loop.
        """
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        payload = RunRequest(
            input=input, session_id=session_id, max_iterations=max_iterations
        ).to_dict()

        def worker() -> None:
            try:
                req = urllib.request.Request(
                    self.url + "/run/stream",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8").strip()
                        if line.startswith("data:"):
                            event = json.loads(line[len("data:") :].strip())
                            loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as exc:  # surfaced to the consumer
                loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

        threading.Thread(target=worker, daemon=True).start()

        while True:
            item = await queue.get()
            if item is _STREAM_DONE:
                break
            if isinstance(item, dict) and "__error__" in item:
                raise A2AError(item["__error__"])
            yield item

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


class A2AClient:
    """A strictly A2A-compliant client (v0.3.0 JSON-RPC).

    Talks to *any* A2A-compliant agent — not just synapse servers. Its
    :meth:`as_tool` turns a remote standards-compliant agent into a synapse
    tool, so a local agent can delegate across the A2A ecosystem.
    """

    def __init__(self, url: str, *, timeout: float = 60.0) -> None:
        self.url = url.rstrip("/")
        self.timeout = timeout
        self._rpc_id = 0

    def _next_id(self) -> int:
        self._rpc_id += 1
        return self._rpc_id

    def _rpc(self, method: str, params: dict) -> dict:
        from . import spec  # noqa: F401 (kept for symmetry / future use)

        body = json.dumps(
            {"jsonrpc": "2.0", "id": self._next_id(), "method": method, "params": params}
        ).encode("utf-8")
        req = urllib.request.Request(
            self.url + "/",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                envelope = json.loads(resp.read())
        except urllib.error.URLError as exc:
            raise A2AError(f"A2A request to {self.url} failed: {exc}") from exc
        if "error" in envelope:
            err = envelope["error"]
            raise A2AError(f"A2A error {err.get('code')}: {err.get('message')}")
        return envelope["result"]

    def card(self):
        from .spec import WELL_KNOWN_PATH, AgentCard

        try:
            with urllib.request.urlopen(self.url + WELL_KNOWN_PATH, timeout=self.timeout) as resp:
                return AgentCard.from_dict(json.loads(resp.read()))
        except urllib.error.URLError as exc:
            raise A2AError(f"failed to fetch agent card: {exc}") from exc

    def send_message(self, text: str, *, context_id: str | None = None) -> dict:
        """Send a message (``message/send``); returns the resulting Task dict."""
        from .spec import Message

        msg = Message.user_text(text)
        if context_id:
            msg.context_id = context_id
        return self._rpc("message/send", {"message": msg.to_dict()})

    def ask(self, text: str) -> str:
        """Send a message and return the agent's final text answer."""
        task = self.send_message(text)
        status = task.get("status", {})
        message = status.get("message") or {}
        parts = message.get("parts", [])
        text_out = "".join(p.get("text", "") for p in parts if p.get("kind") == "text")
        if text_out:
            return text_out
        for artifact in task.get("artifacts", []):
            for p in artifact.get("parts", []):
                if p.get("kind") == "text":
                    return p.get("text", "")
        return ""

    def get_task(self, task_id: str) -> dict:
        return self._rpc("tasks/get", {"id": task_id})

    def cancel_task(self, task_id: str) -> dict:
        return self._rpc("tasks/cancel", {"id": task_id})

    async def astream_message(
        self, text: str, *, context_id: str | None = None
    ) -> "AsyncIterator[dict]":
        """Stream a remote run (``message/stream``), yielding A2A result objects
        (Task, status-update, artifact-update)."""
        from .spec import Message

        msg = Message.user_text(text)
        if context_id:
            msg.context_id = context_id
        payload = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "message/stream",
            "params": {"message": msg.to_dict()},
        }
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()

        def worker() -> None:
            try:
                req = urllib.request.Request(
                    self.url + "/",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    for raw in resp:
                        line = raw.decode("utf-8").strip()
                        if line.startswith("data:"):
                            env = json.loads(line[len("data:") :].strip())
                            loop.call_soon_threadsafe(queue.put_nowait, env)
            except Exception as exc:  # surfaced to consumer
                loop.call_soon_threadsafe(queue.put_nowait, {"__error__": str(exc)})
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _STREAM_DONE)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = await queue.get()
            if item is _STREAM_DONE:
                break
            if isinstance(item, dict) and "__error__" in item:
                raise A2AError(item["__error__"])
            if "error" in item:
                raise A2AError(f"A2A stream error: {item['error']}")
            yield item["result"]

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """Wrap this remote A2A agent as a synapse tool (cross-ecosystem A2A)."""
        card = self.card()
        client = self
        tool_name = name or f"ask_{card.name}".replace(" ", "_")

        async def _delegate(input: str) -> str:
            return await asyncio.to_thread(client.ask, input)

        _delegate.__name__ = tool_name
        return Tool(
            name=tool_name,
            description=description or card.description or f"Delegate to A2A agent {card.name}.",
            parameters={
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Task for the remote A2A agent."}
                },
                "required": ["input"],
            },
            func=_delegate,
        )
