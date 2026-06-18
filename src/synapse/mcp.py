"""MCP adapter: expose Model Context Protocol server tools as synapse tools.

MCP is the de-facto standard for connecting agents to external capabilities.
This adapter wraps an MCP client session so its tools become ordinary synapse
:class:`~synapse.tool.Tool` objects an agent can call. The ``mcp`` package is
an optional dependency, imported lazily — install ``synapse[mcp]`` to use a
real transport. The adapter itself only needs a session object exposing
``list_tools()`` and ``call_tool(name, arguments)``.
"""

from __future__ import annotations

from typing import Any

from .tool import Tool


def _result_to_text(result: Any) -> str:
    """Flatten an MCP ``call_tool`` result into plain text."""
    content = getattr(result, "content", result)
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else str(block))
    return "\n".join(parts)


def tool_from_mcp(mcp_tool: Any, session: Any) -> Tool:
    """Wrap a single MCP tool descriptor as a synapse :class:`Tool`."""
    name = getattr(mcp_tool, "name", None) or mcp_tool["name"]
    description = getattr(mcp_tool, "description", None) or ""
    schema = (
        getattr(mcp_tool, "inputSchema", None)
        or getattr(mcp_tool, "input_schema", None)
        or {"type": "object", "properties": {}, "required": []}
    )

    async def _call(**kwargs: Any) -> str:
        result = await session.call_tool(name, kwargs)
        return _result_to_text(result)

    _call.__name__ = name
    return Tool(name=name, description=description, parameters=schema, func=_call)


async def tools_from_session(session: Any) -> list[Tool]:
    """List an MCP session's tools and wrap each one.

    ``session.list_tools()`` may return a list or an object with a ``.tools``
    attribute (as the official MCP client does).
    """
    listed = await session.list_tools()
    mcp_tools = getattr(listed, "tools", listed)
    return [tool_from_mcp(t, session) for t in mcp_tools]
