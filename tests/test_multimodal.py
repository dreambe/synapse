"""Tests for multimodal content: images in input and tool results."""

from __future__ import annotations

from synapse import Agent, ImageBlock, ScriptedModel, TextBlock, tool
from synapse.messages import ToolResultBlock, message_from_dict
from synapse.models.base import Model, ModelResponse
from synapse.messages import Message


class CaptureModel(Model):
    """Records the messages it was given, then returns a fixed answer."""

    def __init__(self, answer: str = "ok") -> None:
        self.answer = answer
        self.seen: list = []

    async def generate(self, *, system, messages, tools):
        self.seen.append([m.to_dict() for m in messages])
        return ModelResponse(Message("assistant", [TextBlock(self.answer)]))


# -- serialization ----------------------------------------------------------


def test_image_block_base64_wire_shape():
    img = ImageBlock.from_base64("aGVsbG8=", "image/png")
    assert img.to_dict() == {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
    }


def test_image_block_url_wire_shape():
    img = ImageBlock.from_url("https://example.com/cat.png")
    assert img.to_dict() == {
        "type": "image",
        "source": {"type": "url", "url": "https://example.com/cat.png"},
    }


def test_tool_result_multimodal_content_serializes():
    block = ToolResultBlock(
        "t1", [TextBlock("here is the chart"), ImageBlock.from_base64("abc", "image/png")]
    )
    d = block.to_dict()
    assert isinstance(d["content"], list)
    assert d["content"][0]["type"] == "text"
    assert d["content"][1]["type"] == "image"


def test_multimodal_message_roundtrip():
    msg = Message("user", [TextBlock("look"), ImageBlock.from_url("http://x/y.png")])
    back = message_from_dict(msg.to_dict())
    assert isinstance(back.content[1], ImageBlock)
    assert back.content[1].url == "http://x/y.png"

    tr = ToolResultBlock("t1", [ImageBlock.from_base64("zzz", "image/jpeg")])
    back_tr = message_from_dict(Message("user", [tr]).to_dict())
    assert isinstance(back_tr.content[0].content[0], ImageBlock)


# -- multimodal input -------------------------------------------------------


async def test_agent_accepts_image_input():
    model = CaptureModel("described")
    agent = Agent("vision", model=model)
    result = await agent.arun(
        [TextBlock("What is in this image?"), ImageBlock.from_base64("aW1n", "image/png")]
    )
    assert result.output == "described"
    user_msg = model.seen[0][0]
    kinds = [p["type"] for p in user_msg["content"]]
    assert kinds == ["text", "image"]
    assert user_msg["content"][1]["source"]["data"] == "aW1n"


# -- multimodal tool results ------------------------------------------------


async def test_tool_can_return_image():
    @tool
    def render_chart() -> ImageBlock:
        """Render a chart as a PNG."""
        return ImageBlock.from_base64("Y2hhcnQ=", "image/png")

    model = ScriptedModel([[("render_chart", {})], "here is your chart"])
    agent = Agent("charter", model=model, tools=[render_chart])
    result = await agent.arun("make a chart")
    assert result.output == "here is your chart"

    # the tool_result fed back to the model carried the image
    tool_results = [
        b
        for m in result.messages
        for b in m.content
        if isinstance(b, ToolResultBlock)
    ]
    assert tool_results and isinstance(tool_results[0].content, list)
    assert isinstance(tool_results[0].content[0], ImageBlock)


async def test_a2a_image_filepart_maps_to_input():
    from synapse.a2a import A2ADispatcher
    from synapse.a2a.spec import FilePart
    from synapse.a2a.spec import Message as A2AMessage
    from synapse.a2a.spec import TextPart

    model = CaptureModel("saw it")
    disp = A2ADispatcher(Agent("v", model=model))
    a2a_msg = A2AMessage(
        role="user",
        parts=[
            TextPart("describe"),
            FilePart(name="x.png", mime_type="image/png", bytes="aW1n"),
        ],
    )
    resp = await disp.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "message/send",
         "params": {"message": a2a_msg.to_dict()}}
    )
    assert resp["result"]["status"]["state"] == "completed"
    # the agent received a text + image input
    user_msg = model.seen[0][0]
    assert [p["type"] for p in user_msg["content"]] == ["text", "image"]
