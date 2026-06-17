"""The lingua franca of synapse: provider-neutral messages and content blocks.

Every model backend converts to and from these types, so the rest of the
framework — the run loop, tools, A2A transport — never has to know which LLM
produced a turn.
"""

from __future__ import annotations

import base64
import mimetypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Union


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
class ImageBlock:
    """An image, by base64 data (+ media type) or by URL.

    Appears in user input and in tool results (multimodal). Mirrors the
    Anthropic image content block on the wire.
    """

    data: Optional[str] = None
    media_type: Optional[str] = None
    url: Optional[str] = None
    type: str = field(default="image", init=False)

    def to_dict(self) -> dict:
        if self.url:
            source = {"type": "url", "url": self.url}
        else:
            source = {
                "type": "base64",
                "media_type": self.media_type or "image/png",
                "data": self.data or "",
            }
        return {"type": "image", "source": source}

    @classmethod
    def from_base64(cls, data: str, media_type: str = "image/png") -> "ImageBlock":
        return cls(data=data, media_type=media_type)

    @classmethod
    def from_url(cls, url: str) -> "ImageBlock":
        return cls(url=url)

    @classmethod
    def from_file(cls, path: str | Path) -> "ImageBlock":
        p = Path(path)
        media_type = mimetypes.guess_type(p.name)[0] or "image/png"
        data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
        return cls(data=data, media_type=media_type)


@dataclass
class DocumentBlock:
    """A document (PDF, text, …), by base64 data, URL, or inline text.

    Appears in user input and tool results. Mirrors the Anthropic document
    content block; other backends map it as best they can.
    """

    data: Optional[str] = None
    media_type: Optional[str] = None
    url: Optional[str] = None
    text: Optional[str] = None
    title: Optional[str] = None
    type: str = field(default="document", init=False)

    def to_dict(self) -> dict:
        if self.url:
            source = {"type": "url", "url": self.url}
        elif self.text is not None:
            source = {"type": "text", "media_type": "text/plain", "data": self.text}
        else:
            source = {
                "type": "base64",
                "media_type": self.media_type or "application/pdf",
                "data": self.data or "",
            }
        d: dict = {"type": "document", "source": source}
        if self.title:
            d["title"] = self.title
        return d

    @classmethod
    def from_base64(cls, data: str, media_type: str = "application/pdf", *, title: str | None = None):
        return cls(data=data, media_type=media_type, title=title)

    @classmethod
    def from_url(cls, url: str, *, title: str | None = None) -> "DocumentBlock":
        return cls(url=url, title=title)

    @classmethod
    def from_text(cls, text: str, *, title: str | None = None) -> "DocumentBlock":
        return cls(text=text, title=title)

    @classmethod
    def from_file(cls, path: str | Path, *, title: str | None = None) -> "DocumentBlock":
        p = Path(path)
        media_type = mimetypes.guess_type(p.name)[0] or "application/pdf"
        if media_type.startswith("text/"):
            return cls(text=p.read_text(), title=title or p.name)
        data = base64.standard_b64encode(p.read_bytes()).decode("ascii")
        return cls(data=data, media_type=media_type, title=title or p.name)


@dataclass
class ToolResultBlock:
    """The result of executing a tool, fed back to the model.

    ``content`` is either a string or a list of content blocks (e.g. text +
    images) for multimodal tool results.
    """

    tool_use_id: str
    content: "Union[str, list[Any]]"
    is_error: bool = False
    type: str = field(default="tool_result", init=False)

    def to_dict(self) -> dict:
        if isinstance(self.content, str):
            content: object = self.content
        else:
            content = [b.to_dict() for b in self.content]
        return {
            "type": "tool_result",
            "tool_use_id": self.tool_use_id,
            "content": content,
            "is_error": self.is_error,
        }


Block = Union[TextBlock, ImageBlock, DocumentBlock, ToolUseBlock, ToolResultBlock]


@dataclass
class Message:
    """One conversational turn, made of one or more content blocks.

    ``role`` is either ``"user"`` or ``"assistant"``. A string passed as
    ``content`` is normalized into a single :class:`TextBlock`.
    """

    role: str
    content: "Union[str, list[Any]]"

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
        blocks = [TextBlock(self.content)] if isinstance(self.content, str) else self.content
        return {"role": self.role, "content": [b.to_dict() for b in blocks]}


def _block_from_dict(data: dict) -> Block:
    kind = data.get("type")
    if kind == "text":
        return TextBlock(text=data["text"])
    if kind == "image":
        src = data.get("source", {})
        if src.get("type") == "url":
            return ImageBlock(url=src.get("url"))
        return ImageBlock(data=src.get("data"), media_type=src.get("media_type"))
    if kind == "document":
        src = data.get("source", {})
        title = data.get("title")
        if src.get("type") == "url":
            return DocumentBlock(url=src.get("url"), title=title)
        if src.get("type") == "text":
            return DocumentBlock(text=src.get("data"), title=title)
        return DocumentBlock(
            data=src.get("data"), media_type=src.get("media_type"), title=title
        )
    if kind == "tool_use":
        return ToolUseBlock(id=data["id"], name=data["name"], input=data.get("input", {}))
    if kind == "tool_result":
        raw = data.get("content", "")
        content = [_block_from_dict(b) for b in raw] if isinstance(raw, list) else raw
        return ToolResultBlock(
            tool_use_id=data["tool_use_id"],
            content=content,
            is_error=data.get("is_error", False),
        )
    raise ValueError(f"unknown content block type: {kind!r}")


def message_from_dict(data: dict) -> Message:
    """Rebuild a :class:`Message` from its serialized form."""
    return Message(role=data["role"], content=[_block_from_dict(b) for b in data["content"]])
