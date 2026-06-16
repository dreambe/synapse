"""synapse — a composable agent framework.

Build an agent once; call it anywhere. The same :class:`Agent` can be invoked
in-process, exposed as a tool to another agent, or served over HTTP for
agent-to-agent (A2A) communication.

Quick start::

    from synapse import Agent, tool

    @tool
    def add(a: int, b: int) -> int:
        "Add two integers."
        return a + b

    agent = Agent("calculator", instructions="You do arithmetic.", tools=[add])
    print(agent.run("What is 19 * 23, then plus 5?").output)
"""

from __future__ import annotations

from .agent import Agent
from .checkpoint import Checkpointer, FileCheckpointer, InMemoryCheckpointer
from .context import Compactor, select_tools
from .cost import PRICES, Cost, CostTracker, ModelPrice, estimate_cost
from .errors import (
    A2AError,
    ConfigurationError,
    ModelError,
    RegistryError,
    SynapseError,
    ToolError,
)
from .guardrails import (
    Guardrail,
    GuardrailViolation,
    apply_guardrails,
    block_keywords,
    max_length,
    redact,
)
from .memory import FileMemory, InMemoryMemory, Memory, memory_tools
from .messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .models import (
    AnthropicModel,
    EchoModel,
    Model,
    ModelResponse,
    RetryModel,
    ScriptedModel,
)
from .observability import (
    CollectingHooks,
    CompositeHooks,
    Hooks,
    Usage,
)
from .registry import AgentRegistry
from .router import ModelRouter, Router
from .runtime import (
    ApprovalDecision,
    RunContext,
    RunResult,
    RunTimeout,
    Session,
    Verdict,
    arun_agent,
    arun_stream,
    run_agent,
)
from .streaming import RunComplete, RunEvent, TextDelta, ToolCall, ToolOutput
from .team import Blackboard, Team
from .tool import Tool, tool

__version__ = "0.3.0"

__all__ = [
    # core
    "Agent",
    "AgentRegistry",
    "Tool",
    "tool",
    "Message",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "RunResult",
    "RunContext",
    "RunTimeout",
    "Session",
    "run_agent",
    "arun_agent",
    "arun_stream",
    # streaming
    "RunEvent",
    "TextDelta",
    "ToolCall",
    "ToolOutput",
    "RunComplete",
    # models
    "Model",
    "ModelResponse",
    "AnthropicModel",
    "RetryModel",
    "EchoModel",
    "ScriptedModel",
    # observability
    "Hooks",
    "CollectingHooks",
    "CompositeHooks",
    "Usage",
    # memory
    "Memory",
    "InMemoryMemory",
    "FileMemory",
    "memory_tools",
    # guardrails
    "Guardrail",
    "GuardrailViolation",
    "apply_guardrails",
    "block_keywords",
    "max_length",
    "redact",
    # verification / approval
    "Verdict",
    "ApprovalDecision",
    # routing
    "Router",
    "ModelRouter",
    # context engineering
    "Compactor",
    "select_tools",
    # cost
    "Cost",
    "CostTracker",
    "ModelPrice",
    "PRICES",
    "estimate_cost",
    # checkpointing
    "Checkpointer",
    "InMemoryCheckpointer",
    "FileCheckpointer",
    # teams
    "Team",
    "Blackboard",
    # errors
    "SynapseError",
    "ConfigurationError",
    "ModelError",
    "ToolError",
    "RegistryError",
    "A2AError",
    "__version__",
]
