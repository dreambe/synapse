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
from .benchmark import (
    Benchmark,
    DimensionScore,
    Probe,
    ProbeResult,
    Scorecard,
    ScorecardDiff,
    agent_probes,
    compare,
    synapse_benchmark,
)
from .checkpoint import Checkpointer, FileCheckpointer, InMemoryCheckpointer
from .context import Compactor, select_tools
from .errors import (
    A2AError,
    ConfigurationError,
    ModelError,
    RegistryError,
    SynapseError,
    ToolError,
)
from .harness import DEFAULT_INSTRUCTIONS, EDITABLE_SURFACES, Harness, default_harness
from .human import CallbackChannel, HumanChannel, human_tool
from .journal import ExecutionJournal, FileJournal, InMemoryJournal
from .plan import Plan, PlanStep, plan_tools
from .recording import (
    FileRunStore,
    InMemoryRunStore,
    RunRecord,
    RunRecorder,
    RunStore,
)
from .results import FileResultStore, InMemoryResultStore, ResultStore
from .verifiers import command_verifier
from .selfharness import (
    EvidencePack,
    EvolveResult,
    HarnessEdit,
    PromotionRecord,
    classify_failure,
    evolve,
    mine_weaknesses,
    model_proposer,
)
from .evaluation import (
    Case,
    CaseResult,
    Report,
    aevaluate,
    contains,
    equals,
    evaluate,
    llm_judge,
    matches,
)
from .guardrails import (
    Guardrail,
    GuardrailViolation,
    apply_guardrails,
    block_keywords,
    max_length,
    redact,
)
from .memory import (
    Embedder,
    FileMemory,
    FileNamespace,
    HashingEmbedder,
    InMemoryMemory,
    InMemoryNamespace,
    Memory,
    MemoryNamespace,
    OpenAIEmbedder,
    VectorMemory,
    VectorNamespace,
    memory_tools,
)
from .messages import (
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from .monitor import (
    ActivityEvent,
    Monitor,
    RunView,
    classify_tool,
    monitor_app,
)
from .models import (
    AnthropicModel,
    EchoModel,
    Model,
    ModelResponse,
    OpenAIModel,
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
from .sandbox import SandboxResult, code_execution_tool, run_python
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
from .skill import Skill, load_skills, skill_tools
from .streaming import RunComplete, RunEvent, TextDelta, ToolCall, ToolOutput
from .structured import parse_output, validate_json
from .team import Blackboard, Team
from .tool import Tool, tool
from .tracing import OTelHooks, Span, TracingHooks

__version__ = "0.3.0"

__all__ = [
    # core
    "Agent",
    "AgentRegistry",
    "Tool",
    "tool",
    "Message",
    "TextBlock",
    "ImageBlock",
    "DocumentBlock",
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
    "OpenAIModel",
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
    "VectorMemory",
    "Embedder",
    "HashingEmbedder",
    "OpenAIEmbedder",
    "memory_tools",
    # memory isolation (per tenant/user)
    "MemoryNamespace",
    "InMemoryNamespace",
    "FileNamespace",
    "VectorNamespace",
    # sandbox
    "run_python",
    "code_execution_tool",
    "SandboxResult",
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
    # skills
    "Skill",
    "load_skills",
    "skill_tools",
    # structured outputs
    "parse_output",
    "validate_json",
    # evaluation
    "Case",
    "CaseResult",
    "Report",
    "evaluate",
    "aevaluate",
    "contains",
    "equals",
    "matches",
    "llm_judge",
    # benchmark (versioned evaluation standard)
    "Benchmark",
    "Probe",
    "ProbeResult",
    "Scorecard",
    "ScorecardDiff",
    "DimensionScore",
    "compare",
    "agent_probes",
    "synapse_benchmark",
    # run monitor (observability data plane + dashboard)
    "Monitor",
    "ActivityEvent",
    "RunView",
    "classify_tool",
    "monitor_app",
    # tracing
    "TracingHooks",
    "OTelHooks",
    "Span",
    # human-in-the-loop assistance
    "HumanChannel",
    "CallbackChannel",
    "human_tool",
    # plan / decomposition
    "Plan",
    "PlanStep",
    "plan_tools",
    # context offloading
    "ResultStore",
    "InMemoryResultStore",
    "FileResultStore",
    # idempotent execution
    "ExecutionJournal",
    "InMemoryJournal",
    "FileJournal",
    # outcome verification
    "command_verifier",
    # run records (fact source)
    "RunRecord",
    "RunStore",
    "InMemoryRunStore",
    "FileRunStore",
    "RunRecorder",
    # self-harness
    "Harness",
    "default_harness",
    "DEFAULT_INSTRUCTIONS",
    "EDITABLE_SURFACES",
    "HarnessEdit",
    "EvidencePack",
    "PromotionRecord",
    "EvolveResult",
    "evolve",
    "mine_weaknesses",
    "classify_failure",
    "model_proposer",
    # context engineering
    "Compactor",
    "select_tools",
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
