"""Governance: rate limits, quotas, and concurrency caps for agents-as-services.

Serving an agent to many tenants needs admission control that is **structural
and per-scope**, not best-effort. Three composable primitives:

- :class:`RateLimiter` — an async token bucket (sustained ``rate`` per second,
  burst up to ``capacity``). ``try_acquire`` is non-blocking; ``acquire`` waits.
- :class:`Quota` — a cumulative budget (e.g. tokens or calls) that is consumed
  and, optionally, refilled per period. Hard ceiling, not a smoother.
- :class:`Governor` — ties a concurrency cap + rate limit + quota together and
  hands out an **isolated set per scope** (tenant/user), mirroring how
  :class:`~synapse.memory.MemoryNamespace` isolates memory. ``async with
  governor.admit(scope)`` is the one call a server makes per request.

Scope keys must come from authn, never from user-controllable input — the same
rule as tenant memory isolation.

The clock is injectable (``now=``) so behavior is deterministic in tests.
:class:`QuotaExceeded` / :class:`RateLimited` are raised when admission fails in
non-blocking mode.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional

from .errors import SynapseError


class RateLimited(SynapseError):
    """Raised when a non-blocking rate-limit acquisition fails."""


class QuotaExceeded(SynapseError):
    """Raised when a scope's cumulative quota is exhausted."""


class RateLimiter:
    """An async token bucket: ``rate`` tokens/sec, bursting up to ``capacity``."""

    def __init__(
        self,
        rate: float,
        *,
        capacity: Optional[float] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be positive")
        self.rate = rate
        self.capacity = capacity if capacity is not None else rate
        self._now = now
        self._tokens = self.capacity
        self._updated = now()
        self._lock = asyncio.Lock()

    def _replenish(self) -> None:
        t = self._now()
        elapsed = t - self._updated
        if elapsed > 0:
            self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
            self._updated = t

    async def try_acquire(self, cost: float = 1.0) -> bool:
        """Take ``cost`` tokens if available right now; never blocks."""
        async with self._lock:
            self._replenish()
            if self._tokens >= cost:
                self._tokens -= cost
                return True
            return False

    async def acquire(self, cost: float = 1.0) -> None:
        """Wait until ``cost`` tokens are available, then take them."""
        while True:
            async with self._lock:
                self._replenish()
                if self._tokens >= cost:
                    self._tokens -= cost
                    return
                deficit = cost - self._tokens
            await asyncio.sleep(deficit / self.rate)

    @property
    def available(self) -> float:
        self._replenish()
        return self._tokens


class Quota:
    """A cumulative budget over a period (e.g. 1e6 tokens/day). Hard ceiling."""

    def __init__(
        self,
        limit: float,
        *,
        period: Optional[float] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self.limit = limit
        self.period = period
        self._now = now
        self._used = 0.0
        self._window_start = now()

    def _maybe_reset(self) -> None:
        if self.period is not None and self._now() - self._window_start >= self.period:
            self._used = 0.0
            self._window_start = self._now()

    @property
    def used(self) -> float:
        self._maybe_reset()
        return self._used

    @property
    def remaining(self) -> float:
        return max(0.0, self.limit - self.used)

    def would_exceed(self, cost: float) -> bool:
        return self.used + cost > self.limit

    def consume(self, cost: float) -> None:
        """Record ``cost`` against the budget, raising if it would exceed."""
        self._maybe_reset()
        if self._used + cost > self.limit:
            raise QuotaExceeded(
                f"quota exhausted: {self._used:.0f}+{cost:.0f} > {self.limit:.0f}"
            )
        self._used += cost


class _Admission:
    """Async context manager that holds a concurrency slot for the scope."""

    def __init__(self, sem: Optional[asyncio.Semaphore]) -> None:
        self._sem = sem

    async def __aenter__(self) -> "_Admission":
        if self._sem is not None:
            await self._sem.acquire()
        return self

    async def __aexit__(self, *exc: object) -> None:
        if self._sem is not None:
            self._sem.release()


class Governor:
    """Per-scope admission control: concurrency + rate + quota, isolated by scope.

    ``admit(scope)`` is an async context manager: it consumes a rate token,
    charges the quota, and holds a concurrency slot for the duration. Raises
    :class:`RateLimited` / :class:`QuotaExceeded` when ``block=False``.
    """

    def __init__(
        self,
        *,
        max_concurrency: Optional[int] = None,
        rate: Optional[float] = None,
        burst: Optional[float] = None,
        quota: Optional[float] = None,
        quota_period: Optional[float] = None,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._max_concurrency = max_concurrency
        self._rate = rate
        self._burst = burst
        self._quota = quota
        self._quota_period = quota_period
        self._now = now
        self._sems: dict[str, asyncio.Semaphore] = {}
        self._limiters: dict[str, RateLimiter] = {}
        self._quotas: dict[str, Quota] = {}

    def limiter(self, scope: str) -> Optional[RateLimiter]:
        if self._rate is None:
            return None
        if scope not in self._limiters:
            self._limiters[scope] = RateLimiter(self._rate, capacity=self._burst, now=self._now)
        return self._limiters[scope]

    def quota(self, scope: str) -> Optional[Quota]:
        if self._quota is None:
            return None
        if scope not in self._quotas:
            self._quotas[scope] = Quota(self._quota, period=self._quota_period, now=self._now)
        return self._quotas[scope]

    def _sem(self, scope: str) -> Optional[asyncio.Semaphore]:
        if self._max_concurrency is None:
            return None
        if scope not in self._sems:
            self._sems[scope] = asyncio.Semaphore(self._max_concurrency)
        return self._sems[scope]

    def admit(self, scope: str, *, cost: float = 1.0, block: bool = True) -> _Admission:
        """Admit one unit of work for ``scope`` (use as ``async with``)."""
        # The gating happens eagerly in __aenter__ via a coroutine wrapper.
        sem = self._sem(scope)
        limiter = self.limiter(scope)
        quota = self.quota(scope)

        class _Gated(_Admission):
            async def __aenter__(_self) -> "_Admission":  # noqa: N805
                if limiter is not None:
                    if block:
                        await limiter.acquire(cost)
                    elif not await limiter.try_acquire(cost):
                        raise RateLimited(f"rate limit exceeded for scope {scope!r}")
                if quota is not None:
                    quota.consume(cost)  # raises QuotaExceeded
                return await super().__aenter__()

        return _Gated(sem)
