"""Tests for signed, retrying A2A push notifications."""

from __future__ import annotations

import http.server
import json
import threading
import time

from synapse import Agent, ScriptedModel
from synapse.a2a import A2ADispatcher
from synapse.a2a import push
from synapse.a2a.spec import Message


class _OK:
    def close(self) -> None:  # mimic urlopen response
        pass


def test_signature_roundtrip():
    body = b'{"a":1}'
    ts = str(int(time.time()))
    sig = push._signature(body, "secret", ts)
    assert push.verify_signature(body, "secret", ts, sig)
    assert not push.verify_signature(body, "wrong-secret", ts, sig)


def test_signature_rejects_stale_timestamp():
    body = b"{}"
    old = str(int(time.time()) - 10_000)
    sig = push._signature(body, "secret", old)
    assert not push.verify_signature(body, "secret", old, sig, max_age_seconds=300)


def test_deliver_signs_payload():
    captured: dict = {}

    def opener(req, timeout=None):
        captured["headers"] = {k.lower(): v for k, v in req.header_items()}
        captured["body"] = req.data
        return _OK()

    ok = push.deliver(
        {"url": "http://x", "token": "secret"},
        {"task": "t"},
        opener=opener,
        sleep=lambda *_: None,
    )
    assert ok
    h = captured["headers"]
    assert h["x-a2a-notification-token"] == "secret"
    assert push.verify_signature(
        captured["body"], "secret", h["x-a2a-timestamp"], h["x-a2a-signature"]
    )


def test_deliver_retries_then_succeeds():
    calls = {"n": 0}

    def opener(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 3:
            raise ConnectionError("receiver down")
        return _OK()

    ok = push.deliver({"url": "http://x"}, {"a": 1}, opener=opener, sleep=lambda *_: None)
    assert ok and calls["n"] == 3


def test_deliver_gives_up_after_retries():
    def opener(req, timeout=None):
        raise ConnectionError("down")

    ok = push.deliver(
        {"url": "http://x"}, {"a": 1}, retries=2, opener=opener, sleep=lambda *_: None
    )
    assert ok is False


async def test_push_notification_end_to_end():
    received: dict = {}

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            received["body"] = self.rfile.read(n)
            received["sig"] = self.headers.get("X-A2A-Signature")
            received["ts"] = self.headers.get("X-A2A-Timestamp")
            received["token"] = self.headers.get("X-A2A-Notification-Token")
            self.send_response(200)
            self.end_headers()

    srv = http.server.HTTPServer(("127.0.0.1", 0), Handler)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        disp = A2ADispatcher(Agent("p", model=ScriptedModel(["pushed result"])))
        await disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tasks/pushNotificationConfig/set",
                "params": {
                    "taskId": "T1",
                    "pushNotificationConfig": {
                        "url": f"http://127.0.0.1:{port}",
                        "token": "sekret",
                    },
                },
            }
        )
        await disp.handle(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "message/send",
                "params": {"message": {**Message.user_text("go").to_dict(), "taskId": "T1"}},
            }
        )
        assert received.get("body") is not None
        assert received["token"] == "sekret"
        assert push.verify_signature(received["body"], "sekret", received["ts"], received["sig"])
        assert json.loads(received["body"])["status"]["state"] == "completed"
    finally:
        srv.shutdown()
