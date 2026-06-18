"""Streaming primitives.

Streaming is treated as the *underlying* execution model, not a bolt-on: the
run loop drives a stream of events internally, and the non-streaming ``run``
simply consumes that stream to its final result. This module defines the event
types that flow out of a run (and the model-level chunk a backend yields).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from .models.base import ModelResponse
    from .runtime import RunResult


@dataclass
class TextDelta:
    """An incremental chunk of assistant text."""

    text: str
    type: str = "text_delta"


@dataclass
class ToolCall:
    """The model asked to call a tool (emitted before execution)."""

    id: str
    name: str
    input: dict
    type: str = "tool_call"


@dataclass
class ToolOutput:
    """A tool finished executing (emitted after execution)."""

    id: str
    name: str
    content: str
    is_error: bool
    type: str = "tool_output"


@dataclass
class Steered:
    """Operator guidance was injected into the running agent (mid-run steering)."""

    text: str
    type: str = "steered"


@dataclass
class RunComplete:
    """Terminal event carrying the final :class:`~synapse.runtime.RunResult`."""

    result: "RunResult"
    type: str = "run_complete"


# Public events yielded by Agent.astream / arun_stream.
RunEvent = Union[TextDelta, ToolCall, ToolOutput, Steered, RunComplete]


@dataclass
class ModelStreamEnd:
    """Final model-level chunk: the fully assembled assistant turn."""

    response: "ModelResponse"
    type: str = "model_stream_end"


# What a Model.stream() yields: text deltas, then exactly one ModelStreamEnd.
ModelChunk = Union[TextDelta, ModelStreamEnd]
