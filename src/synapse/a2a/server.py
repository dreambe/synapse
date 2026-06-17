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
import queue as _queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from .protocol import RunRequest, RunResponse, run_event_to_dict

if TYPE_CHECKING:
    from ..agent import Agent

CARD_PATH = "/.well-known/agent.json"


def _make_handler(agent: "Agent", sessions: dict, loop: asyncio.AbstractEventLoop):
    from ..runtime import RunContext, Session

    _STREAM_DONE = object()

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

        def _read_request(self) -> RunRequest:
            length = int(self.headers.get("Content-Length", 0))
            data = json.loads(self.rfile.read(length) or b"{}")
            return RunRequest.from_dict(data)

        def do_POST(self) -> None:  # noqa: N802
            if self.path == "/run/stream":
                self._handle_stream()
                return
            if self.path != "/run":
                self._send_json(404, {"error": "not found"})
                return
            try:
                request = self._read_request()
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

        def _handle_stream(self) -> None:
            try:
                request = self._read_request()
            except (ValueError, KeyError) as exc:
                self._send_json(400, {"error": f"bad request: {exc}"})
                return

            session = None
            if request.session_id is not None:
                session = sessions.setdefault(request.session_id, Session())

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            bridge: _queue.Queue = _queue.Queue()

            async def consume() -> None:
                try:
                    async for ev in agent.astream(
                        request.input,
                        session=session,
                        context=RunContext(max_iterations=request.max_iterations),
                    ):
                        bridge.put(run_event_to_dict(ev))
                finally:
                    bridge.put(_STREAM_DONE)

            future = asyncio.run_coroutine_threadsafe(consume(), loop)
            while True:
                item = bridge.get()
                if item is _STREAM_DONE:
                    break
                self.wfile.write(f"data: {json.dumps(item)}\n\n".encode("utf-8"))
                self.wfile.flush()
            future.result()  # surface any exception from the run

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
        self.host, self.port = str(self._httpd.server_address[0]), int(self._httpd.server_address[1])
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
