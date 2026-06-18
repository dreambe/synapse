"""Tests for governance: rate limit, quota, concurrency (deterministic clock)."""

from __future__ import annotations

import asyncio

import pytest

from synapse import Governor, Quota, QuotaExceeded, RateLimited, RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_rate_limiter_burst_then_refill():
    clock = FakeClock()
    rl = RateLimiter(rate=2.0, capacity=2.0, now=clock)

    async def go():
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is False  # burst exhausted
        clock.advance(0.5)  # 0.5s * 2/s = 1 token
        assert await rl.try_acquire() is True
        assert await rl.try_acquire() is False

    asyncio.run(go())


def test_quota_hard_ceiling_and_reset():
    clock = FakeClock()
    q = Quota(limit=10, period=60, now=clock)
    q.consume(7)
    assert q.remaining == 3
    with pytest.raises(QuotaExceeded):
        q.consume(5)
    clock.advance(61)  # new window
    q.consume(5)
    assert q.used == 5


def test_governor_admit_charges_rate_and_quota_per_scope():
    clock = FakeClock()
    gov = Governor(rate=1.0, burst=1.0, quota=2, now=clock)

    async def go():
        async with gov.admit("alice", block=False):
            pass
        # alice is rate-limited now (burst spent), but bob is isolated
        with pytest.raises(RateLimited):
            async with gov.admit("alice", block=False):
                pass
        async with gov.admit("bob", block=False):
            pass

    asyncio.run(go())


def test_governor_quota_exhaustion():
    gov = Governor(quota=2)

    async def go():
        async with gov.admit("alice"):
            pass
        async with gov.admit("alice"):
            pass
        with pytest.raises(QuotaExceeded):
            async with gov.admit("alice"):
                pass

    asyncio.run(go())


def test_governor_concurrency_cap_blocks():
    gov = Governor(max_concurrency=1)
    order = []

    async def worker(tag, release):
        async with gov.admit("alice"):
            order.append(f"enter-{tag}")
            await release.wait()
            order.append(f"exit-{tag}")

    async def go():
        r1 = asyncio.Event()
        t1 = asyncio.create_task(worker("1", r1))
        await asyncio.sleep(0)  # let t1 acquire the only slot
        t2 = asyncio.create_task(worker("2", asyncio.Event()))
        await asyncio.sleep(0)
        assert order == ["enter-1"]  # t2 is blocked on the semaphore
        r1.set()
        await t1
        t2.cancel()

    asyncio.run(go())
