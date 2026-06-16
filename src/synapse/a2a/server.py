"""Serve an agent over HTTP — the zero-dependency path.

Built on the standard library only, so a synapse agent becomes a network
service with no extra dependencies::

    serve(my_agent, port=8080)

Requests are handled on worker threads (concurrent inbound) but every agent
run is submitted to a single shared event loop, so async sessions and tools
behave exactly as they do under the ASGI server. For the highest throughput
(one loop, no thread ceiling) run the ASGI app under uvicorn instead — see
:mod:`synapse.a2a.asgi`.

Endpoints:
    GET  /.well-known/agent.json   → the agent card
    GET  /health                   → liveness probe
    POST /run                      → run the agent on a task
"""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from .protocol import RunRequest, RunResponse

if TYPE_CHECKING:
    from ..agent import Agent

CARD_PATH = "/.well-known/agent.json"


def _make_handler(agent: "Agent", sessions: dict, loop: asyncio.AbstractEventLoop):
    from ..runtime import Session

    class A2AHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args) -> None:  # quiet by default
            pass

        def _send_json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 (stdlib naming)
            if self.path == CARD_PATH:
                host = self.headers.get("Host")
                url = f"http://{host}" if host else None
                self._send_json(200, agent.card(url=url).to_dict())
            elif self.path == "/health":
                self._send_json(200, {"status": "ok", "agent": agent.name})
            else:
                self._send_json(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/run":
                self._send_json(404, {"error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", 0))
                data = json.loads(self.rfile.read(length) or b"{}")
                request = RunRequest.from_dict(data)
            except (ValueError, KeyError) as exc:
                self._send_json(400, {"error": f"bad request: {exc}"})
                return

            session = None
            if request.session_id is not None:
                session = sessions.setdefault(request.session_id, Session())

            # Submit the async run to the shared loop and wait for the result.
            future = asyncio.run_coroutine_threadsafe(
                agent.arun(
                    request.input,
                    max_iterations=request.max_iterations,
                    session=session,
                ),
                loop,
            )
            result = future.result()
            response = RunResponse(
                output=result.output,
                agent=result.agent,
                iterations=result.iterations,
                stop_reason=result.stop_reason,
                session_id=request.session_id,
            )
            self._send_json(200, response.to_dict())

    return A2AHandler


class AgentServer:
    """A running HTTP server wrapping a single agent."""

    def __init__(self, agent: "Agent", host: str = "127.0.0.1", port: int = 8080) -> None:
        self.agent = agent
        self._sessions: dict = {}
        self._loop = asyncio.new_event_loop()
        self._loop_thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._loop_thread.start()
        self._httpd = ThreadingHTTPServer(
            (host, port), _make_handler(agent, self._sessions, self._loop)
        )
        self.host, self.port = self._httpd.server_address[0], self._httpd.server_address[1]
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start_background(self) -> "AgentServer":
        """Serve in a daemon thread and return immediately (handy for tests)."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def serve_forever(self) -> None:
        """Block and serve until interrupted."""
        try:
            self._httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._loop.call_soon_threadsafe(self._loop.stop)


def serve(agent: "Agent", *, host: str = "127.0.0.1", port: int = 8080) -> None:
    """Serve ``agent`` over HTTP, blocking until interrupted."""
    server = AgentServer(agent, host=host, port=port)
    print(f"synapse: serving agent {agent.name!r} at {server.url}")
    print(f"  card:   {server.url}{CARD_PATH}")
    server.serve_forever()
