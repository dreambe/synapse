"""Tests for the v0.2 capabilities — all offline, no API key."""

from __future__ import annotations

import pytest

from synapse import (
    Agent,
    Blackboard,
    CollectingHooks,
    Compactor,
    GuardrailViolation,
    InMemoryCheckpointer,
    InMemoryMemory,
    Router,
    ScriptedModel,
    Team,
    Usage,
    Verdict,
    block_keywords,
    redact,
    select_tools,
    tool,
)
from synapse.models.base import Model, ModelResponse
from synapse.messages import Message, TextBlock


# -- helpers ----------------------------------------------------------------


class UsageToolModel(Model):
    """Always asks for the `noop` tool and reports fixed usage per call."""

    def __init__(self, per_call: int = 100) -> None:
        self.per_call = per_call

    async def generate(self, *, system, messages, tools):
        from synapse.messages import ToolUseBlock

        return ModelResponse(
            Message("assistant", [ToolUseBlock(id="t", name="noop", input={})]),
            stop_reason="tool_use",
            usage=Usage(self.per_call // 2, self.per_call // 2),
        )


class FailingModel(Model):
    def __init__(self) -> None:
        self.calls = 0

    async def generate(self, *, system, messages, tools):
        self.calls += 1
        raise RuntimeError("boom")


class FlakyModel(Model):
    """Fails `fail_times` then answers."""

    def __init__(self, fail_times: int, answer: str) -> None:
        self.fail_times = fail_times
        self.answer = answer
        self.calls = 0

    async def generate(self, *, system, messages, tools):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient")
        return ModelResponse(Message("assistant", [TextBlock(self.answer)]))


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@tool
def noop() -> str:
    """Do nothing."""
    return "ok"


# -- observability ----------------------------------------------------------


async def test_hooks_capture_lifecycle():
    hooks = CollectingHooks()
    agent = Agent("h", model=ScriptedModel([[("add", {"a": 1, "b": 2})], "done"]), tools=[add])
    await agent.arun("go", hooks=hooks)
    names = [e[0] for e in hooks.events]
    assert names[0] == "run_start"
    assert "tool_start" in names and "tool_end" in names
    assert names[-1] == "run_end"


async def test_usage_accumulates_and_budget_stops():
    agent = Agent("b", model=UsageToolModel(per_call=100), tools=[noop])
    result = await agent.arun("go", token_budget=150, max_iterations=10)
    assert result.stop_reason == "budget_exceeded"
    assert result.usage.total_tokens >= 150


# -- memory -----------------------------------------------------------------


async def test_memory_remember_and_recall():
    mem = InMemoryMemory()
    agent = Agent(
        "m",
        model=ScriptedModel(
            [
                [("remember", {"fact": "the sky is blue"})],
                [("recall", {"query": "sky"})],
                "answered",
            ]
        ),
        memory=mem,
    )
    result = await agent.arun("remember and recall")
    assert result.output == "answered"
    assert "the sky is blue" in await mem.all()
    assert {"remember", "recall"} <= set(agent.tool_map)


# -- guardrails -------------------------------------------------------------


async def test_input_guardrail_blocks():
    agent = Agent("g", model=ScriptedModel(["hi"]))
    with pytest.raises(GuardrailViolation):
        await agent.arun("this is forbidden", input_guardrails=[block_keywords(["forbidden"])])


async def test_output_guardrail_redacts():
    agent = Agent("g", model=ScriptedModel(["my password is hunter2"]))
    result = await agent.arun("x", output_guardrails=[redact(r"hunter2")])
    assert result.output == "my password is [redacted]"


# -- approval (HITL) --------------------------------------------------------


async def test_tool_approval_denies():
    @tool(requires_approval=True)
    def danger() -> str:
        """A dangerous action."""
        return "did it"

    agent = Agent("a", model=ScriptedModel([[("danger", {})], "after"]), tools=[danger])
    result = await agent.arun("go", approval=lambda name, inp: False)
    assert result.output == "after"
    denied = [
        b
        for m in result.messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result" and b.is_error
    ]
    assert denied and "denied" in denied[0].content.lower()


async def test_tool_approval_allows():
    calls = []

    @tool(requires_approval=True)
    def act() -> str:
        """Act."""
        calls.append(1)
        return "acted"

    agent = Agent("a", model=ScriptedModel([[("act", {})], "fin"]), tools=[act])
    await agent.arun("go", approval=lambda name, inp: True)
    assert calls == [1]


# -- verifier loop ----------------------------------------------------------


async def test_verifier_iterates_until_pass():
    seen = {"n": 0}

    def verify(output: str) -> Verdict:
        seen["n"] += 1
        return Verdict(passed=seen["n"] >= 2, feedback="try harder")

    agent = Agent("v", model=ScriptedModel(["first try", "second try"]))
    result = await agent.arun("solve", verify=verify, max_verify_rounds=3)
    assert result.output == "second try"
    assert result.stop_reason == "verified"
    assert result.verify_rounds == 2


# -- routing ----------------------------------------------------------------


async def test_router_picks_agent():
    billing = Agent("billing", model=ScriptedModel(["billing handled"]))
    support = Agent("support", model=ScriptedModel(["support handled"]))
    router = Router(default=support)
    router.add_keyword_route(["invoice", "payment"], billing)

    assert router.select("question about my invoice") is billing
    assert router.select("how do I reset") is support
    result = await router.arun("invoice problem")
    assert result.output == "billing handled"


# -- resilience -------------------------------------------------------------


async def test_retry_then_succeed():
    from synapse import RetryModel

    flaky = FlakyModel(fail_times=2, answer="recovered")
    agent = Agent("r", model=RetryModel(flaky, max_retries=2, base_delay=0))
    result = await agent.arun("go")
    assert result.output == "recovered"
    assert flaky.calls == 3


async def test_fallback_model():
    from synapse import RetryModel

    primary = FailingModel()
    agent = Agent(
        "r",
        model=RetryModel(primary, fallbacks=[ScriptedModel(["from fallback"])], max_retries=1, base_delay=0),
    )
    result = await agent.arun("go")
    assert result.output == "from fallback"
    assert primary.calls == 2  # original + one retry before falling back


# -- checkpointing ----------------------------------------------------------


async def test_checkpoint_save_and_resume():
    cp = InMemoryCheckpointer()
    agent = Agent("c", model=ScriptedModel(["first", "second"]))

    await agent.arun("hello", checkpointer=cp, run_id="run-1")
    saved = await cp.load("run-1")
    assert saved is not None and any(m.text == "first" for m in saved)

    # Resume: prior history is restored, then the new input is appended.
    result = await agent.arun("again", checkpointer=cp, run_id="run-1")
    user_inputs = [m.text for m in result.messages if m.role == "user"]
    assert "hello" in user_inputs and "again" in user_inputs


# -- context engineering ----------------------------------------------------


async def test_compactor_summarizes():
    summarizer = ScriptedModel(["A SUMMARY"])
    compactor = Compactor(summarizer, trigger_messages=2, keep_recent=1)
    messages = [Message("user", f"m{i}") for i in range(5)]
    out = await compactor.maybe_compact(messages)
    assert len(out) == 2
    assert "A SUMMARY" in out[0].text


def test_select_tools_ranks_by_relevance():
    picked = select_tools("add numbers", [add, noop], k=1)
    assert picked[0].name == "add"


async def test_tool_search_flow():
    agent = Agent(
        "s",
        model=ScriptedModel(
            [
                [("search_tools", {"query": "add"})],
                [("add", {"a": 2, "b": 3})],
                "5",
            ]
        ),
        tools=[add, noop],
        tool_search=True,
    )
    result = await agent.arun("add 2 and 3")
    assert result.output == "5"
    tool_results = [
        b.content
        for m in result.messages
        for b in m.content
        if getattr(b, "type", None) == "tool_result"
    ]
    assert any("5" == c for c in tool_results)  # add executed


# -- teams ------------------------------------------------------------------


async def test_blackboard_share():
    bb = Blackboard()
    await bb.write("finding", "x=42")
    assert await bb.read("finding") == "x=42"


async def test_team_wires_members_and_blackboard():
    member = Agent("specialist", model=ScriptedModel(["spec result"]), description="A specialist.")
    coordinator = Agent(
        "coord",
        model=ScriptedModel([[("ask_specialist", {"input": "do it"})], "team done"]),
    )
    team = Team(coordinator, [member])
    assert "ask_specialist" in coordinator.tool_map
    assert "post_note" in coordinator.tool_map and "read_notes" in member.tool_map
    result = await team.arun("delegate")
    assert result.output == "team done"


# -- mcp adapter ------------------------------------------------------------


class _FakeMCPTool:
    def __init__(self, name, description, schema):
        self.name = name
        self.description = description
        self.inputSchema = schema


class _FakeContentBlock:
    def __init__(self, text):
        self.text = text


class _FakeMCPResult:
    def __init__(self, text):
        self.content = [_FakeContentBlock(text)]


class _FakeMCPSession:
    def __init__(self):
        self._tools = [
            _FakeMCPTool(
                "echo",
                "Echo the input",
                {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]},
            )
        ]

    async def list_tools(self):
        return self._tools

    async def call_tool(self, name, arguments):
        return _FakeMCPResult(f"{name}:{arguments.get('text')}")


async def test_mcp_tools_from_session():
    from synapse.mcp import tools_from_session

    session = _FakeMCPSession()
    tools = await tools_from_session(session)
    assert len(tools) == 1
    echo = tools[0]
    assert echo.name == "echo"
    assert echo.parameters["required"] == ["text"]
    assert await echo.invoke(text="hi") == "echo:hi"
