"""An ASGI application that serves an agent with native async concurrency.

This is the high-throughput serving path: run it under an ASGI server such as
uvicorn (``pip install synapse[server]``) and a single process handles many
concurrent ``/run`` requests on one event loop — no thread-per-request ceiling.

    from synapse.a2a.asgi import create_app
    app = create_app(my_agent)
    # uvicorn synapse_app:app   (or: synapse serve mymod:agent)

Endpoints mirror the stdlib server:
    GET  /.well-known/agent.json   → the agent card
    GET  /health                   → liveness probe
    POST /run                      → run the agent on a task
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Awaitable, Callable

from ..runtime import RunContext, Session
from .dispatcher import A2ADispatcher
from .protocol import RunRequest, RunResponse, run_event_to_dict
from .server import CARD_PATH
from .spec import WELL_KNOWN_PATH, WELL_KNOWN_PATH_LEGACY

if TYPE_CHECKING:
    from ..agent import Agent


def _base_url(scope: Scope) -> str:
    headers = {k.decode().lower(): v.decode() for k, v in scope.get("headers", [])}
    host = headers.get("host", "localhost")
    scheme = scope.get("scheme", "http")
    return f"{scheme}://{host}"


def create_a2a_app(agent: "Agent"):
    """An ASGI app serving ``agent`` over the **A2A protocol** (v0.3.0).

    Endpoints:
        GET  /.well-known/agent-card.json   → the A2A Agent Card
        POST /                              → JSON-RPC 2.0 (message/send,
             message/stream [SSE], tasks/get, tasks/cancel, tasks/resubscribe,
             tasks/pushNotificationConfig/set,get)
    """
    dispatcher = A2ADispatcher(agent)

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return
        if scope["type"] != "http":  # pragma: no cover
            return

        method, path = scope["method"], scope["path"]
        if method == "GET" and path in (WELL_KNOWN_PATH, WELL_KNOWN_PATH_LEGACY):
            await _send_json(send, 200, dispatcher.agent_card(_base_url(scope)).to_dict())
            return

        if method == "POST":
            try:
                payload = json.loads(await _read_body(receive) or b"{}")
            except ValueError:
                await _send_json(send, 200, {"jsonrpc": "2.0", "id": None,
                                             "error": {"code": -32700, "message": "parse error"}})
                return
            rpc_method = payload.get("method")
            if rpc_method in ("message/stream", "tasks/resubscribe"):
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"content-type", b"text/event-stream"),
                            (b"cache-control", b"no-cache"),
                        ],
                    }
                )
                async for env in dispatcher.stream(payload):
                    frame = f"data: {json.dumps(env)}\n\n".encode()
                    await send({"type": "http.response.body", "body": frame, "more_body": True})
                await send({"type": "http.response.body", "body": b"", "more_body": False})
                return
            await _send_json(send, 200, await dispatcher.handle(payload))
            return

        await _send_json(send, 404, {"error": "not found"})

    return app

Scope = dict[str, Any]
Receive = Callable[[], Awaitable[dict]]
Send = Callable[[dict], Awaitable[None]]


async def _read_body(receive: Receive) -> bytes:
    chunks: list[bytes] = []
    more = True
    while more:
        event = await receive()
        chunks.append(event.get("body", b""))
        more = event.get("more_body", False)
    return b"".join(chunks)


async def _send_json(send: Send, status: int, payload: dict) -> None:
    body = json.dumps(payload).encode("utf-8")
    await send(
        {
            "type": "http.response.start",
            "status": status,
            "headers": [
                (b"content-type", b"application/json"),
                (b"content-length", str(len(body)).encode()),
            ],
        }
    )
    await send({"type": "http.response.body", "body": body})


def create_app(agent: "Agent") -> Callable[[Scope, Receive, Send], Awaitable[None]]:
    """Build an ASGI app that serves ``agent``."""
    sessions: dict[str, Session] = {}

    async def app(scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            while True:
                message = await receive()
                if message["type"] == "lifespan.startup":
                    await send({"type": "lifespan.startup.complete"})
                elif message["type"] == "lifespan.shutdown":
                    await send({"type": "lifespan.shutdown.complete"})
                    return

        if scope["type"] != "http":  # pragma: no cover - websockets etc.
            return

        method, path = scope["method"], scope["path"]

        if method == "GET" and path == CARD_PATH:
            await _send_json(send, 200, agent.card().to_dict())
            return
        if method == "GET" and path == "/health":
            await _send_json(send, 200, {"status": "ok", "agent": agent.name})
            return
        if method == "POST" and path == "/run/stream":
            try:
                data = json.loads(await _read_body(receive) or b"{}")
                request = RunRequest.from_dict(data)
            except (ValueError, KeyError) as exc:
                await _send_json(send, 400, {"error": f"bad request: {exc}"})
                return

            session = None
            if request.session_id is not None:
                session = sessions.setdefault(request.session_id, Session())

            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"text/event-stream"),
                        (b"cache-control", b"no-cache"),
                    ],
                }
            )
            async for ev in agent.astream(
                request.input,
                session=session,
                context=RunContext(max_iterations=request.max_iterations),
            ):
                chunk = f"data: {json.dumps(run_event_to_dict(ev))}\n\n".encode()
                await send({"type": "http.response.body", "body": chunk, "more_body": True})
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            return

        if method == "POST" and path == "/run":
            try:
                data = json.loads(await _read_body(receive) or b"{}")
                request = RunRequest.from_dict(data)
            except (ValueError, KeyError) as exc:
                await _send_json(send, 400, {"error": f"bad request: {exc}"})
                return

            session = None
            if request.session_id is not None:
                session = sessions.setdefault(request.session_id, Session())

            result = await agent.arun(
                request.input,
                max_iterations=request.max_iterations,
                session=session,
            )
            response = RunResponse(
                output=result.output,
                agent=result.agent,
                iterations=result.iterations,
                stop_reason=result.stop_reason,
                session_id=request.session_id,
            )
            await _send_json(send, 200, response.to_dict())
            return

        await _send_json(send, 404, {"error": "not found"})

    return app
