"""Resilience: wrap any backend with retries, backoff, and model fallback.

    model = RetryModel(AnthropicModel(), fallbacks=[AnthropicModel("claude-sonnet-4-6")])

On a *transient* failure the wrapper retries with exponential backoff, then
walks the fallback chain. ``retry_on`` decides what counts as transient — by
default connection/timeout/model errors. A failure outside that set is treated
as permanent and raised immediately, so a bug in your code is not silently
retried 3x. The same shape covers Anthropic's refusal→fallback story: put a
more permissive model later in ``fallbacks``.

Note: ``RetryModel`` streams via the inherited single-chunk path (retry and
fallback apply to the underlying ``generate``); it does not re-stream tokens
through the retry layer. Wrap the model directly when you need native
token streaming without retry.
"""

from __future__ import annotations

import asyncio

from ..errors import ModelError
from .base import Model, ModelResponse

# Transient by default: network hiccups, timeouts, and backend-reported errors.
DEFAULT_RETRYABLE: tuple[type[BaseException], ...] = (
    ModelError,
    ConnectionError,
    TimeoutError,
    asyncio.TimeoutError,
)


class RetryModel(Model):
    """Decorate a :class:`Model` with retry, backoff, and fallbacks."""

    def __init__(
        self,
        model: Model,
        *,
        fallbacks: list[Model] | None = None,
        max_retries: int = 2,
        base_delay: float = 0.5,
        max_delay: float = 8.0,
        retry_on: tuple[type[BaseException], ...] = DEFAULT_RETRYABLE,
    ) -> None:
        self.model = model
        self.fallbacks = list(fallbacks or [])
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.retry_on = retry_on

    async def _attempt(self, backend: Model, **kwargs) -> ModelResponse:
        last_exc: BaseException | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await backend.generate(**kwargs)
            except self.retry_on as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    delay = min(self.base_delay * (2**attempt), self.max_delay)
                    await asyncio.sleep(delay)
            # Non-retryable exceptions propagate immediately (not swallowed).
        assert last_exc is not None
        raise last_exc

    async def generate(self, *, system, messages, tools) -> ModelResponse:
        chain = [self.model, *self.fallbacks]
        errors: list[str] = []
        for backend in chain:
            try:
                return await self._attempt(
                    backend, system=system, messages=messages, tools=tools
                )
            except self.retry_on as exc:
                errors.append(f"{type(backend).__name__}: {exc}")
        raise ModelError("all model backends failed: " + " | ".join(errors))
