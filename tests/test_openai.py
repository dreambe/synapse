"""Tests for the OpenAI-compatible backend (provider-neutral proof)."""

from __future__ import annotations

from synapse import Agent, ImageBlock, OpenAIModel, TextBlock, tool
from synapse.messages import Message, ToolResultBlock, ToolUseBlock
from synapse.models.openai import (
    from_openai_response,
    to_openai_messages,
    to_openai_tools,
)


# -- fakes mimicking the openai SDK response objects ------------------------


class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, id, name, arguments):
        self.id = id
        self.type = "function"
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls or []


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c


class _Resp:
    def __init__(self, message, usage=None):
        self.choices = [type("C", (), {"message": message})()]
        self.usage = usage


# -- message conversion -----------------------------------------------------


def test_system_and_text_message():
    out = to_openai_messages("be brief", [Message("user", "hi")])
    assert out[0] == {"role": "system", "content": "be brief"}
    assert out[1] == {"role": "user", "content": "hi"}


def test_image_parts_become_image_url():
    m = Message("user", [TextBlock("what's this"), ImageBlock.from_base64("aW1n", "image/png")])
    content = to_openai_messages("", [m])[0]["content"]
    assert content[0] == {"type": "text", "text": "what's this"}
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"] == "data:image/png;base64,aW1n"


def test_tool_call_and_result_roundtrip():
    convo = [
        Message("user", "add"),
        Message("assistant", [ToolUseBlock(id="c1", name="add", input={"a": 1})]),
        Message("user", [ToolResultBlock("c1", "2")]),
    ]
    out = to_openai_messages("", convo)
    assistant = next(m for m in out if m["role"] == "assistant")
    assert assistant["tool_calls"][0]["function"]["name"] == "add"
    tool_msg = next(m for m in out if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1" and tool_msg["content"] == "2"


def test_tools_schema():
    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    t = to_openai_tools([add])[0]
    assert t["type"] == "function" and t["function"]["name"] == "add"
    assert "a" in t["function"]["parameters"]["properties"]


# -- response parsing -------------------------------------------------------


def test_parse_text_response():
    mr = from_openai_response(_Resp(_Msg(content="hello"), _Usage(10, 5)))
    assert mr.stop_reason == "end_turn"
    assert mr.message.text == "hello"
    assert mr.usage.input_tokens == 10 and mr.usage.output_tokens == 5


def test_parse_tool_call_response():
    mr = from_openai_response(_Resp(_Msg(tool_calls=[_ToolCall("c1", "add", '{"a": 1, "b": 2}')])))
    assert mr.stop_reason == "tool_use"
    tu = mr.message.tool_uses[0]
    assert tu.name == "add" and tu.input == {"a": 1, "b": 2}


# -- end-to-end with an injected fake client --------------------------------


async def test_openai_model_generate_with_fake_client():
    captured: dict = {}

    class FakeCompletions:
        async def create(self, **kwargs):
            captured.update(kwargs)
            return _Resp(_Msg(content="42"), _Usage(3, 1))

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    agent = Agent("o", model=OpenAIModel("gpt-4o", client=FakeClient()))
    result = await agent.arun("2+2?")
    assert result.output == "42"
    assert result.usage.input_tokens == 3
    assert captured["model"] == "gpt-4o"
    assert captured["messages"][-1] == {"role": "user", "content": "2+2?"}


async def test_openai_model_tool_loop():
    # Two responses: a tool call, then a final answer.
    class FakeCompletions:
        def __init__(self):
            self.n = 0

        async def create(self, **kwargs):
            self.n += 1
            if self.n == 1:
                return _Resp(_Msg(tool_calls=[_ToolCall("c1", "add", '{"a": 2, "b": 2}')]))
            return _Resp(_Msg(content="the sum is 4"))

    class FakeChat:
        completions = FakeCompletions()

    class FakeClient:
        chat = FakeChat()

    @tool
    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    agent = Agent("o", model=OpenAIModel("gpt-4o", client=FakeClient()), tools=[add])
    result = await agent.arun("2+2?")
    assert result.output == "the sum is 4"
