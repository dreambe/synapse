"""Anthropic Claude backend (async, with native token streaming).

Requires the optional ``anthropic`` extra:  ``pip install synapse[anthropic]``.
The async SDK client is constructed lazily so importing this module never fails
when the dependency or an API key is absent. A single client multiplexes many
concurrent requests, which is exactly what high-throughput agent serving needs.
"""

from __future__ import annotations

from typing import Any, AsyncIterator

from ..errors import ModelError
from ..messages import Message, TextBlock, ToolUseBlock
from ..observability import Usage
from ..streaming import ModelChunk, ModelStreamEnd, TextDelta
from ..tool import Tool
from .base import Model, ModelResponse

# Default to the most capable Claude model. Override per-instance if needed.
DEFAULT_MODEL = "claude-opus-4-8"


class AnthropicModel(Model):
    """Claude via the official ``anthropic`` async SDK."""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        max_tokens: int = 4096,
        thinking: bool = True,
        client: Any | None = None,
        **client_kwargs: Any,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.thinking = thinking
        self._client = client
        self._client_kwargs = client_kwargs

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover - import guard
                raise ModelError(
                    "the anthropic package is required for AnthropicModel; "
                    "install it with `pip install synapse[anthropic]`"
                ) from exc
            self._client = anthropic.AsyncAnthropic(**self._client_kwargs)
        return self._client

    def _build_request(
        self, system: str, messages: list[Message], tools: list[Tool]
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [m.to_dict() for m in messages],
        }
        if system:
            request["system"] = system
        if tools:
            request["tools"] = [t.to_schema() for t in tools]
        if self.thinking:
            request["thinking"] = {"type": "adaptive"}
        return request

    @staticmethod
    def _to_response(raw: Any) -> ModelResponse:
        content: list = []
        for block in raw.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
            # thinking blocks are intentionally dropped from the persisted turn
        stop = "tool_use" if raw.stop_reason == "tool_use" else "end_turn"
        usage = None
        if getattr(raw, "usage", None) is not None:
            usage = Usage(
                input_tokens=getattr(raw.usage, "input_tokens", 0) or 0,
                output_tokens=getattr(raw.usage, "output_tokens", 0) or 0,
            )
        return ModelResponse(
            message=Message(role="assistant", content=content), stop_reason=stop, usage=usage
        )

    async def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResponse:
        client = self._get_client()
        try:
            raw = await client.messages.create(**self._build_request(system, messages, tools))
        except Exception as exc:  # pragma: no cover - network/runtime
            raise ModelError(f"Anthropic request failed: {exc}") from exc
        return self._to_response(raw)

    async def stream(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
    ) -> AsyncIterator[ModelChunk]:
        client = self._get_client()
        request = self._build_request(system, messages, tools)
        try:
            async with client.messages.stream(**request) as stream:
                async for text in stream.text_stream:
                    if text:
                        yield TextDelta(text)
                final = await stream.get_final_message()
        except Exception as exc:  # pragma: no cover - network/runtime
            raise ModelError(f"Anthropic stream failed: {exc}") from exc
        yield ModelStreamEnd(self._to_response(final))
