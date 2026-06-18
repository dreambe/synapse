"""OpenAI-compatible backend.

Works with any provider that speaks the OpenAI Chat Completions API — OpenAI,
Azure OpenAI, Together, Groq, Mistral, vLLM, Ollama, etc. — by pointing
``base_url`` at it. Requires the optional ``openai`` extra:
``pip install synapse[openai]``.

This is the proof that synapse is provider-neutral: the framework speaks its own
message format (:mod:`synapse.messages`) and each backend translates. To add a
provider with no OpenAI-compatible endpoint, implement :class:`~synapse.models.base.Model`.

Multimodal note: text and images map cleanly to OpenAI content parts. Documents
map by best effort — inline ``text`` documents become text; binary documents
(e.g. PDF) are summarized as a placeholder, since OpenAI-compatible document
support is provider-specific.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..errors import ModelError
from ..messages import (
    DocumentBlock,
    ImageBlock,
    Message,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
)
from ..observability import Usage
from ..streaming import ModelChunk, ModelStreamEnd, TextDelta
from ..tool import Tool
from .base import Model, ModelResponse


def _image_url(block: ImageBlock) -> str:
    if block.url:
        return block.url
    return f"data:{block.media_type or 'image/png'};base64,{block.data or ''}"


def _parts_to_openai(blocks: list) -> Any:
    """Render content blocks as OpenAI content (a string if all text)."""
    if all(isinstance(b, TextBlock) for b in blocks):
        return "".join(b.text for b in blocks)
    parts: list[dict] = []
    for b in blocks:
        if isinstance(b, TextBlock):
            parts.append({"type": "text", "text": b.text})
        elif isinstance(b, ImageBlock):
            parts.append({"type": "image_url", "image_url": {"url": _image_url(b)}})
        elif isinstance(b, DocumentBlock):
            if b.text is not None:
                parts.append({"type": "text", "text": b.text})
            else:
                label = b.title or b.media_type or "document"
                parts.append({"type": "text", "text": f"[document: {label} — not inlined]"})
    return parts


def _tool_result_to_openai(block: ToolResultBlock) -> str:
    if isinstance(block.content, str):
        return block.content
    chunks = []
    for b in block.content:
        chunks.append(b.text if isinstance(b, TextBlock) else f"[{getattr(b, 'type', 'block')}]")
    return " ".join(chunks)


def to_openai_messages(system: str, messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    for m in messages:
        if m.role == "user":
            tool_results = [b for b in m.content if isinstance(b, ToolResultBlock)]
            for tr in tool_results:
                out.append(
                    {"role": "tool", "tool_call_id": tr.tool_use_id, "content": _tool_result_to_openai(tr)}
                )
            others = [b for b in m.content if not isinstance(b, ToolResultBlock)]
            if others or not tool_results:
                out.append({"role": "user", "content": _parts_to_openai(others)})
        else:  # assistant
            tool_uses = [b for b in m.content if isinstance(b, ToolUseBlock)]
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_uses:
                msg["tool_calls"] = [
                    {
                        "id": tu.id,
                        "type": "function",
                        "function": {"name": tu.name, "arguments": json.dumps(tu.input)},
                    }
                    for tu in tool_uses
                ]
            out.append(msg)
    return out


def to_openai_tools(tools: list[Tool]) -> list[dict]:
    return [
        {
            "type": "function",
            "function": {"name": t.name, "description": t.description, "parameters": t.parameters},
        }
        for t in tools
    ]


def from_openai_response(resp: Any) -> ModelResponse:
    choice = resp.choices[0]
    message = choice.message
    content: list = []
    if getattr(message, "content", None):
        content.append(TextBlock(message.content))
    tool_calls = getattr(message, "tool_calls", None) or []
    for tc in tool_calls:
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (ValueError, TypeError):
            args = {}
        content.append(ToolUseBlock(id=tc.id, name=tc.function.name, input=args))
    stop = "tool_use" if tool_calls else "end_turn"
    usage = None
    u = getattr(resp, "usage", None)
    if u is not None:
        usage = Usage(
            input_tokens=getattr(u, "prompt_tokens", 0) or 0,
            output_tokens=getattr(u, "completion_tokens", 0) or 0,
        )
    return ModelResponse(message=Message("assistant", content), stop_reason=stop, usage=usage)


class OpenAIModel(Model):
    """Any OpenAI Chat Completions-compatible endpoint."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 4096,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_tokens = max_tokens
        self._client = client
        self._client_kwargs = client_kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover - import guard
                raise ModelError(
                    "the openai package is required for OpenAIModel; "
                    "install it with `pip install synapse[openai]`"
                ) from exc
            kwargs = dict(self._client_kwargs)
            if self.base_url:
                kwargs["base_url"] = self.base_url
            if self.api_key:
                kwargs["api_key"] = self.api_key
            self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    async def generate(self, *, system, messages, tools) -> ModelResponse:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": to_openai_messages(system, messages),
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        try:
            resp = await client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - network/runtime
            raise ModelError(f"OpenAI-compatible request failed: {exc}") from exc
        return from_openai_response(resp)

    async def stream(self, *, system, messages, tools) -> AsyncIterator[ModelChunk]:
        client = self._get_client()
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": to_openai_messages(system, messages),
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            payload["tools"] = to_openai_tools(tools)
        try:
            resp = await client.chat.completions.create(**payload)
        except Exception as exc:  # pragma: no cover - network/runtime
            raise ModelError(f"OpenAI-compatible stream failed: {exc}") from exc

        text_parts: list[str] = []
        tool_calls: dict[int, dict] = {}
        usage = None
        async for chunk in resp:
            if getattr(chunk, "usage", None) is not None:
                usage = chunk.usage
            choices = getattr(chunk, "choices", None) or []
            if not choices:
                continue
            delta = choices[0].delta
            content = getattr(delta, "content", None)
            if content:
                text_parts.append(content)
                yield TextDelta(content)
            for tc in getattr(delta, "tool_calls", None) or []:
                slot = tool_calls.setdefault(tc.index, {"id": None, "name": None, "args": ""})
                if getattr(tc, "id", None):
                    slot["id"] = tc.id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["args"] += fn.arguments

        content_blocks: list = []
        text = "".join(text_parts)
        if text:
            content_blocks.append(TextBlock(text))
        for idx in sorted(tool_calls):
            slot = tool_calls[idx]
            try:
                args = json.loads(slot["args"] or "{}")
            except (ValueError, TypeError):
                args = {}
            content_blocks.append(
                ToolUseBlock(id=slot["id"] or f"call_{idx}", name=slot["name"] or "", input=args)
            )
        stop = "tool_use" if tool_calls else "end_turn"
        out_usage = None
        if usage is not None:
            out_usage = Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        yield ModelStreamEnd(
            ModelResponse(Message("assistant", content_blocks), stop_reason=stop, usage=out_usage)
        )
