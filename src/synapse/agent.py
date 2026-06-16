"""The Agent: the central abstraction of synapse.

An agent bundles instructions, a set of tools, and a model backend. It can be
run directly (sync or async), exposed as a tool to *another* agent (in-process
A2A), served over HTTP, or described by an agent card.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .models import Model, default_model
from .runtime import RunResult, Session, arun_agent, run_agent
from .tool import Tool

if TYPE_CHECKING:
    from .a2a.protocol import AgentCard


class Agent:
    """A callable, composable unit of agentic behavior."""

    def __init__(
        self,
        name: str,
        *,
        instructions: str = "",
        description: str = "",
        model: Model | None = None,
        tools: list[Tool] | None = None,
        version: str = "0.1.0",
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.description = description or instructions.split("\n")[0][:200]
        self.model = model or default_model()
        self.tools: list[Tool] = list(tools or [])
        self.version = version

    @property
    def tool_map(self) -> dict[str, Tool]:
        return {t.name: t for t in self.tools}

    # -- composition ---------------------------------------------------------

    def add_tool(self, t: Tool) -> "Agent":
        """Register a tool and return self for chaining."""
        self.tools.append(t)
        return self

    def tool(self, func: Callable | None = None, **kwargs):
        """Decorator that builds a :class:`Tool` and attaches it to this agent."""
        from .tool import tool as _tool

        def wrap(f: Callable) -> Tool:
            built = _tool(f, **kwargs)
            self.add_tool(built)
            return built

        return wrap(func) if func is not None else wrap

    # -- invocation ----------------------------------------------------------

    async def arun(
        self,
        user_input: str,
        *,
        max_iterations: int = 12,
        session: Session | None = None,
    ) -> RunResult:
        """Run this agent to completion (async). Use under any event loop."""
        return await arun_agent(
            self, user_input, max_iterations=max_iterations, session=session
        )

    def run(
        self,
        user_input: str,
        *,
        max_iterations: int = 12,
        session: Session | None = None,
    ) -> RunResult:
        """Synchronous convenience wrapper around :meth:`arun`."""
        return run_agent(
            self, user_input, max_iterations=max_iterations, session=session
        )

    # -- agent-to-agent ------------------------------------------------------

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent as a tool so another agent can delegate to it.

        This is the in-process form of A2A: a coordinator lists the returned
        tool alongside its own. The delegate is async, so a coordinator that
        calls several sub-agents in one turn runs them concurrently.
        """
        agent = self
        tool_name = name or f"ask_{self.name}"

        async def _delegate(input: str) -> str:
            return (await agent.arun(input)).output

        _delegate.__name__ = tool_name
        return Tool(
            name=tool_name,
            description=(
                description
                or self.description
                or f"Delegate a task to the {self.name} agent."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "The task or question to hand to the agent.",
                    }
                },
                "required": ["input"],
            },
            func=_delegate,
        )

    def card(self, url: str | None = None) -> "AgentCard":
        """Describe this agent as an :class:`AgentCard` for discovery."""
        from .a2a.protocol import AgentCard

        return AgentCard(
            name=self.name,
            description=self.description,
            version=self.version,
            skills=[t.name for t in self.tools],
            url=url,
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, tools={[t.name for t in self.tools]})"
