"""Guardrails: validate or transform agent input and output.

A guardrail is a callable ``(text) -> text`` that may pass the text through,
return a sanitized version, or raise :class:`GuardrailViolation` to block the
run. Input guardrails run before the loop; output guardrails run on the final
answer. Guardrails may be sync or async.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable, Union

from ._util import maybe_await
from .errors import SynapseError

Guardrail = Callable[[str], Union[str, Awaitable[str]]]


class GuardrailViolation(SynapseError):
    """Raised by a guardrail to block a run."""


async def apply_guardrails(text: str, guardrails: list[Guardrail]) -> str:
    """Run ``text`` through each guardrail in order, threading the result."""
    for guard in guardrails:
        text = await maybe_await(guard(text))
    return text


def block_keywords(keywords: list[str], *, message: str = "blocked content") -> Guardrail:
    """Reject text containing any of ``keywords`` (case-insensitive)."""
    lowered = [k.lower() for k in keywords]

    def guard(text: str) -> str:
        haystack = text.lower()
        for kw in lowered:
            if kw in haystack:
                raise GuardrailViolation(f"{message}: matched {kw!r}")
        return text

    return guard


def max_length(limit: int) -> Guardrail:
    """Reject text longer than ``limit`` characters."""

    def guard(text: str) -> str:
        if len(text) > limit:
            raise GuardrailViolation(f"text exceeds {limit} characters")
        return text

    return guard


def redact(pattern: str, *, replacement: str = "[redacted]") -> Guardrail:
    """Replace every match of ``pattern`` with ``replacement`` (transforming)."""
    compiled = re.compile(pattern)

    def guard(text: str) -> str:
        return compiled.sub(replacement, text)

    return guard
