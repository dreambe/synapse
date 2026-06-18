"""Tools: plain Python callables exposed to an agent with a JSON schema.

Decorate a function with :func:`tool` and synapse derives the input schema from
its signature and type hints, and the description from its docstring. Tools may
be ordinary functions *or* ``async def`` coroutines — the run loop awaits the
latter and offloads the former to a thread, so a blocking tool never stalls the
event loop.
"""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Callable, get_type_hints

from .errors import ToolError

_PY_TO_JSON: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    dict: "object",
    list: "array",
}


@dataclass
class Tool:
    """A callable plus the metadata a model needs to call it."""

    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]
    # When True, the run loop asks the configured approval callback before
    # executing this tool (human-in-the-loop / policy gating).
    requires_approval: bool = False
    # Declared side effect, used by permission policies. "read" tools are safe
    # to run in read-only / plan mode; the safe default is "write" (mutating).
    side_effect: str = "write"

    @property
    def is_async(self) -> bool:
        return inspect.iscoroutinefunction(self.func)

    def to_schema(self) -> dict:
        """Render the Anthropic-style tool schema."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }

    async def invoke(self, **kwargs: Any) -> Any:
        """Execute the tool without blocking the event loop.

        Async tools are awaited directly; sync tools run in a worker thread.
        """
        if self.is_async:
            return await self.func(**kwargs)
        return await asyncio.to_thread(lambda: self.func(**kwargs))

    def __call__(self, **kwargs: Any) -> Any:
        """Direct call. For sync tools returns the value; for async tools
        returns the coroutine (await it yourself)."""
        return self.func(**kwargs)


def _json_type(annotation: Any) -> str:
    return _PY_TO_JSON.get(annotation, "string")


def _build_tool(
    func: Callable[..., Any],
    name: str | None,
    description: str | None,
    requires_approval: bool = False,
    side_effect: str = "write",
) -> Tool:
    sig = inspect.signature(func)
    try:
        hints = get_type_hints(func)
    except Exception:  # pragma: no cover - exotic annotations
        hints = {}

    properties: dict[str, dict] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise ToolError(f"tool {func.__name__!r} may not use *args/**kwargs")
        annotation = hints.get(param_name, str)
        properties[param_name] = {"type": _json_type(annotation)}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    doc = inspect.getdoc(func) or ""
    resolved_description = description or doc.split("\n\n")[0].strip() or func.__name__

    return Tool(
        name=name or func.__name__,
        description=resolved_description,
        parameters={
            "type": "object",
            "properties": properties,
            "required": required,
        },
        func=func,
        requires_approval=requires_approval,
        side_effect=side_effect,
    )


def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    requires_approval: bool = False,
    side_effect: str = "write",
) -> Any:
    """Turn a function (sync or ``async def``) into a :class:`Tool`.

    Usable bare (``@tool``) or with arguments (``@tool(name=...)``)::

        @tool
        def add(a: int, b: int) -> int:
            "Add two integers."
            return a + b

        @tool
        async def fetch(url: str) -> str:
            "Fetch a URL."
            ...
    """

    def wrap(f: Callable[..., Any]) -> Tool:
        return _build_tool(f, name, description, requires_approval, side_effect)

    if func is not None:
        return wrap(func)
    return wrap
