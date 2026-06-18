"""Small async helpers shared across the framework."""

from __future__ import annotations

import inspect
from typing import Any


async def maybe_await(value: Any) -> Any:
    """Await ``value`` if it is awaitable, otherwise return it as-is.

    Lets callbacks (hooks, approval, guardrails, verifiers) be written as
    either plain functions or coroutines.
    """
    if inspect.isawaitable(value):
        return await value
    return value
