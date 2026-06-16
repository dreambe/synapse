"""The Agent: the central abstraction of synapse.

An agent bundles instructions, tools, and a model backend. It can be run
directly (sync, async, or streaming), exposed as a tool to another agent
(in-process A2A), served over HTTP, or described by an agent card.

The canonical configuration object is :class:`~synapse.runtime.RunContext`;
the keyword arguments on :meth:`arun` are convenience sugar over it. ``run``
and ``astream`` are thin forwarders, so there is one option surface to learn.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator, Callable

from .guardrails import Guardrail
from .memory import Memory, memory_tools
from .models import Model, default_model
from .observability import Hooks
from .runtime import (
    ApprovalCallback,
    RunContext,
    RunResult,
    Session,
    Verifier,
    arun_agent,
    arun_stream,
    run_sync,
)
from .streaming import RunEvent
from .tool import Tool

if TYPE_CHECKING:
    from .a2a.protocol import AgentCard
    from .checkpoint import Checkpointer
    from .context import Compactor


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
        memory: Memory | None = None,
        tool_search: bool = False,
        version: str = "0.1.0",
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.description = description or instructions.split("\n")[0][:200]
        self.model = model or default_model()
        self.tools: list[Tool] = list(tools or [])
        self.memory = memory
        self.tool_search = tool_search
        self.version = version
        if memory is not None:
            self.tools.extend(memory_tools(memory))

    @property
    def tool_map(self) -> dict[str, Tool]:
        return {t.name: t for t in self.tools}

    # -- composition ---------------------------------------------------------

    def add_tool(self, t: Tool) -> "Agent":
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

    def _context(self, max_iterations: int, context: RunContext | None, **opts) -> RunContext:
        if context is not None:
            return context
        return RunContext(
            max_iterations=max_iterations,
            hooks=opts.get("hooks"),
            approval=opts.get("approval"),
            verify=opts.get("verify"),
            max_verify_rounds=opts.get("max_verify_rounds") or 3,
            token_budget=opts.get("token_budget"),
            timeout=opts.get("timeout"),
            tool_timeout=opts.get("tool_timeout"),
            max_parallel_tools=opts.get("max_parallel_tools"),
            input_guardrails=opts.get("input_guardrails") or [],
            output_guardrails=opts.get("output_guardrails") or [],
            compactor=opts.get("compactor"),
            checkpointer=opts.get("checkpointer"),
            run_id=opts.get("run_id"),
            pricing=opts.get("pricing"),
            tool_search=(
                self.tool_search if opts.get("tool_search") is None else opts["tool_search"]
            ),
        )

    async def arun(
        self,
        user_input: str,
        *,
        max_iterations: int = 12,
        session: Session | None = None,
        context: RunContext | None = None,
        hooks: Hooks | None = None,
        approval: ApprovalCallback | None = None,
        verify: Verifier | None = None,
        max_verify_rounds: int = 3,
        token_budget: int | None = None,
        timeout: float | None = None,
        tool_timeout: float | None = None,
        max_parallel_tools: int | None = None,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
        compactor: "Compactor | None" = None,
        checkpointer: "Checkpointer | None" = None,
        run_id: str | None = None,
        pricing: dict | None = None,
        tool_search: bool | None = None,
    ) -> RunResult:
        """Run this agent to completion (async). The canonical, typed entry point."""
        ctx = self._context(
            max_iterations,
            context,
            hooks=hooks,
            approval=approval,
            verify=verify,
            max_verify_rounds=max_verify_rounds,
            token_budget=token_budget,
            timeout=timeout,
            tool_timeout=tool_timeout,
            max_parallel_tools=max_parallel_tools,
            input_guardrails=input_guardrails,
            output_guardrails=output_guardrails,
            compactor=compactor,
            checkpointer=checkpointer,
            run_id=run_id,
            pricing=pricing,
            tool_search=tool_search,
        )
        return await arun_agent(self, user_input, session=session, context=ctx)

    def run(self, user_input: str, **kwargs) -> RunResult:
        """Synchronous wrapper — accepts the same keyword arguments as :meth:`arun`."""
        return run_sync(self.arun(user_input, **kwargs))

    def astream(
        self,
        user_input: str,
        *,
        session: Session | None = None,
        max_iterations: int = 12,
        context: RunContext | None = None,
        **kwargs,
    ) -> AsyncIterator[RunEvent]:
        """Run this agent and yield events as they happen (text deltas, tool
        calls, tool outputs), ending with ``RunComplete``. Same options as
        :meth:`arun`."""
        ctx = self._context(max_iterations, context, **kwargs)
        return arun_stream(self, user_input, session=session, context=ctx)

    # -- agent-to-agent ------------------------------------------------------

    def as_tool(self, *, name: str | None = None, description: str | None = None) -> Tool:
        """Expose this agent as a tool so another agent can delegate to it.

        The delegate is async, so a coordinator that calls several sub-agents
        in one turn runs them concurrently.
        """
        agent = self
        tool_name = name or f"ask_{self.name}"

        async def _delegate(input: str) -> str:
            return (await agent.arun(input)).output

        _delegate.__name__ = tool_name
        return Tool(
            name=tool_name,
            description=(
                description or self.description or f"Delegate a task to the {self.name} agent."
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
