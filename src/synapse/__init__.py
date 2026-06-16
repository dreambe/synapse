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
from .errors import (
    A2AError,
    ConfigurationError,
    ModelError,
    RegistryError,
    SynapseError,
    ToolError,
)
from .messages import Message, TextBlock, ToolResultBlock, ToolUseBlock
from .models import AnthropicModel, EchoModel, Model, ModelResponse, ScriptedModel
from .registry import AgentRegistry
from .runtime import RunResult, Session, arun_agent, run_agent
from .tool import Tool, tool

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentRegistry",
    "Tool",
    "tool",
    "Message",
    "TextBlock",
    "ToolUseBlock",
    "ToolResultBlock",
    "Model",
    "ModelResponse",
    "AnthropicModel",
    "EchoModel",
    "ScriptedModel",
    "RunResult",
    "Session",
    "run_agent",
    "arun_agent",
    "SynapseError",
    "ConfigurationError",
    "ModelError",
    "ToolError",
    "RegistryError",
    "A2AError",
    "__version__",
]
