"""Anthropic Claude backend (async).

Requires the optional ``anthropic`` extra:  ``pip install synapse[anthropic]``.
The async SDK client is constructed lazily so importing this module never fails
when the dependency or an API key is absent. A single client multiplexes many
concurrent requests, which is exactly what high-throughput agent serving needs.
"""

from __future__ import annotations

from typing import Any

from ..errors import ModelError
from ..messages import Message, TextBlock, ToolUseBlock
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

    async def generate(
        self,
        *,
        system: str,
        messages: list[Message],
        tools: list[Tool],
    ) -> ModelResponse:
        client = self._get_client()

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

        try:
            response = await client.messages.create(**request)
        except Exception as exc:  # pragma: no cover - network/runtime
            raise ModelError(f"Anthropic request failed: {exc}") from exc

        content: list = []
        for block in response.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                content.append(ToolUseBlock(id=block.id, name=block.name, input=dict(block.input)))
            # thinking blocks are intentionally dropped from the persisted turn

        stop = "tool_use" if response.stop_reason == "tool_use" else "end_turn"
        return ModelResponse(message=Message(role="assistant", content=content), stop_reason=stop)
