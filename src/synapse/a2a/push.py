"""A2A push notifications: signed, retrying webhook delivery.

When a client registers a push config (``tasks/pushNotificationConfig/set``),
the agent POSTs the terminal task to that webhook. Baseline A2A only requires
fire-and-forget; this adds two things receivers actually need in production:

- **Authenticity** — an HMAC-SHA256 signature over ``timestamp + "." + body``
  (header ``X-A2A-Signature``), plus the client's validation ``token`` echoed in
  ``X-A2A-Notification-Token``. Receivers verify with :func:`verify_signature`.
- **Reliability** — exponential-backoff retries on transient delivery failures.

Network and sleep are injectable so this is fully unit-testable offline.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import urllib.request
from typing import Any, Callable, Optional

DEFAULT_RETRIES = 3


def _signature(body: bytes, secret: str, timestamp: str) -> str:
    mac = hmac.new(secret.encode("utf-8"), timestamp.encode("utf-8") + b"." + body, hashlib.sha256)
    return "sha256=" + mac.hexdigest()


def verify_signature(
    body: bytes,
    secret: str,
    timestamp: str,
    signature: str,
    *,
    max_age_seconds: Optional[int] = 300,
) -> bool:
    """Verify a webhook signature (constant-time), optionally rejecting stale
    timestamps to mitigate replay."""
    if max_age_seconds is not None:
        try:
            if abs(time.time() - int(timestamp)) > max_age_seconds:
                return False
        except (TypeError, ValueError):
            return False
    return hmac.compare_digest(_signature(body, secret, timestamp), signature)


def deliver(
    config: dict,
    payload: dict,
    *,
    retries: int = DEFAULT_RETRIES,
    base_delay: float = 0.2,
    max_delay: float = 5.0,
    timeout: float = 10.0,
    opener: Optional[Callable[..., Any]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> bool:
    """Deliver ``payload`` to ``config['url']`` with signing + retries.

    Returns True on a successful delivery, False if all attempts failed (the
    caller should not raise — push delivery is auxiliary to the run).
    """
    url = config.get("url")
    if not url:
        return False

    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    token = config.get("token")
    if token:
        timestamp = str(int(time.time()))
        headers["X-A2A-Notification-Token"] = token
        headers["X-A2A-Timestamp"] = timestamp
        headers["X-A2A-Signature"] = _signature(body, token, timestamp)

    send = opener or urllib.request.urlopen
    for attempt in range(retries + 1):
        try:
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            resp = send(req, timeout=timeout)
            close = getattr(resp, "close", None)
            if close:
                close()
            return True
        except Exception:  # noqa: BLE001 - any delivery failure is retryable
            if attempt < retries:
                sleep(min(base_delay * (2**attempt), max_delay))
    return False
