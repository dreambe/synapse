"""The lingua franca of synapse: provider-neutral messages and content blocks.

Every model backend converts to and from these types, so the rest of the
framework — the run loop, tools, A2A transport — never has to know which LLM
produced a turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass
class TextBlock:
    """A span of natural-language text."""

    text: str
    type: str = field(default="text", init=False)

    def to_dict(self) -> dict:
        return {"type": "text", "text": self.text}


@dataclass
class ToolUseBlock:
    """A model's request to invoke a tool with structured input."""

    id: str
    name: str
    input: dict
    type: str = field(default="tool_use", init=False)

    def to_dict(self) -> dict:
        return {"type": "tool_use", "id": self.id, "name": self.name, "input": self.input}


@dataclass
class ToolResultBlock:
    """The result of executing a tool, fed back to the model."""

    tool_use_id: str
    content: str
    is_error: bool = False
    type: str = field(default="tool_result", init=False)

    def to_dict(self) -> dict:
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": self.content,
            "is_error": self.is_error,
        }


Block = Union[TextBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class Message:
    """One conversational turn, made of one or more content blocks.

    ``role`` is either ``"user"`` or ``"assistant"``. A string passed as
    ``content`` is normalized into a single :class:`TextBlock`.
    """

    role: str
    content: list[Block]

    def __post_init__(self) -> None:
        if isinstance(self.content, str):
            self.content = [TextBlock(self.content)]

    @property
    def text(self) -> str:
        """Concatenated text of every :class:`TextBlock` in this message."""
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_uses(self) -> list[ToolUseBlock]:
        return [b for b in self.content if isinstance(b, ToolUseBlock)]

    def to_dict(self) -> dict:
        return {"role": self.role, "content": [b.to_dict() for b in self.content]}


def _block_from_dict(data: dict) -> Block:
    kind = data.get("type")
    if kind == "text":
        return TextBlock(text=data["text"])
    if kind == "tool_use":
        return ToolUseBlock(id=data["id"], name=data["name"], input=data.get("input", {}))
    if kind == "tool_result":
        return ToolResultBlock(
            tool_use_id=data["tool_use_id"],
            content=data.get("content", ""),
            is_error=data.get("is_error", False),
        )
    raise ValueError(f"unknown content block type: {kind!r}")


def message_from_dict(data: dict) -> Message:
    """Rebuild a :class:`Message` from its serialized form."""
    return Message(role=data["role"], content=[_block_from_dict(b) for b in data["content"]])
