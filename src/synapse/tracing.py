"""Tracing: turn lifecycle hooks into spans.

:class:`TracingHooks` records in-memory spans (run + each tool call) with no
dependency — inspect ``.spans`` or feed them to any exporter. :class:`OTelHooks`
emits real OpenTelemetry spans (optional ``[otel]`` extra).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

from .observability import Hooks


@dataclass
class Span:
    name: str
    attributes: dict = field(default_factory=dict)
    start: float = 0.0
    end: Optional[float] = None

    @property
    def duration(self) -> Optional[float]:
        return None if self.end is None else self.end - self.start


class TracingHooks(Hooks):
    """Record in-memory spans for the run and each tool call (dependency-free).

    Note: tool spans are keyed by tool name, so two *concurrent* calls to the
    same tool in one turn share a span boundary — fine for tracing, not exact
    timing. For precise per-call spans use :class:`OTelHooks`.
    """

    def __init__(self) -> None:
        self.spans: list[Span] = []
        self._run: Optional[Span] = None
        self._tools: dict[str, Span] = {}

    async def on_run_start(self, agent: str, user_input: str) -> None:
        self._run = Span(f"run:{agent}", {"agent": agent}, time.monotonic())

    async def on_tool_start(self, name: str, tool_input: dict) -> None:
        self._tools[name] = Span(f"tool:{name}", {"tool": name}, time.monotonic())

    async def on_tool_end(self, name: str, result: str, is_error: bool) -> None:
        span = self._tools.pop(name, None)
        if span is not None:
            span.end = time.monotonic()
            span.attributes["is_error"] = is_error
            self.spans.append(span)

    async def on_run_end(self, result: Any) -> None:
        if self._run is not None:
            self._run.end = time.monotonic()
            self._run.attributes.update(
                iterations=result.iterations,
                tokens=result.usage.total_tokens,
                stop_reason=result.stop_reason,
            )
            self.spans.append(self._run)


class OTelHooks(Hooks):
    """Emit OpenTelemetry spans. Requires ``pip install synapse[otel]``."""

    def __init__(self, tracer: Any = None) -> None:
        try:
            from opentelemetry import trace
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "OTelHooks needs OpenTelemetry; install it with `pip install synapse[otel]`"
            ) from exc
        self._tracer = tracer or trace.get_tracer("synapse")
        self._run = None
        self._tools: dict[str, Any] = {}

    async def on_run_start(self, agent: str, user_input: str) -> None:
        self._run = self._tracer.start_span("synapse.run", attributes={"agent": agent})

    async def on_tool_start(self, name: str, tool_input: dict) -> None:
        self._tools[name] = self._tracer.start_span(f"synapse.tool.{name}")

    async def on_tool_end(self, name: str, result: str, is_error: bool) -> None:
        span = self._tools.pop(name, None)
        if span is not None:
            span.set_attribute("is_error", is_error)
            span.end()

    async def on_run_end(self, result: Any) -> None:
        if self._run is not None:
            self._run.set_attribute("iterations", result.iterations)
            self._run.set_attribute("tokens", result.usage.total_tokens)
            self._run.set_attribute("stop_reason", result.stop_reason)
            self._run.end()
