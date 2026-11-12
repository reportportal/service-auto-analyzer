#  Copyright 2026 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#  https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import asyncio

import pytest

from app.coordinator.budget import IntakeBudget


async def _settle() -> None:
    for _ in range(3):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_acquire_within_limits_does_not_wait():
    budget = IntakeBudget(max_items=3, max_bytes=100)

    await asyncio.wait_for(budget.acquire(2, 60), 1)

    assert budget.snapshot() == {"items": 2, "max_items": 3, "bytes": 60, "max_bytes": 100, "waiting": 0}


@pytest.mark.asyncio
async def test_waiters_are_admitted_in_fifo_order():
    budget = IntakeBudget(max_items=3, max_bytes=100)
    await budget.acquire(1, 60)
    order: list[str] = []

    async def take(name: str, items: int, size: int) -> None:
        await budget.acquire(items, size)
        order.append(name)

    big = asyncio.create_task(take("big", 1, 50))
    await _settle()
    # Fits by itself, but must queue behind the waiting big message
    small = asyncio.create_task(take("small", 1, 1))
    await _settle()
    assert order == []
    assert budget.snapshot()["waiting"] == 2

    budget.release(1, 60)
    await asyncio.wait_for(asyncio.gather(big, small), 1)

    assert order == ["big", "small"]
    assert budget.snapshot()["items"] == 2
    assert budget.snapshot()["bytes"] == 51


@pytest.mark.asyncio
async def test_item_limit_blocks():
    budget = IntakeBudget(max_items=2, max_bytes=1000)
    await budget.acquire(2, 2)

    waiter = asyncio.create_task(budget.acquire(1, 1))
    await _settle()
    assert not waiter.done()

    budget.release(1, 1)
    await asyncio.wait_for(waiter, 1)
    assert budget.snapshot()["items"] == 2


@pytest.mark.asyncio
async def test_oversized_message_admitted_only_when_empty():
    budget = IntakeBudget(max_items=3, max_bytes=100)
    await budget.acquire(1, 10)

    oversized = asyncio.create_task(budget.acquire(1, 1000))
    await _settle()
    assert not oversized.done()

    budget.release(1, 10)
    await asyncio.wait_for(oversized, 1)
    assert budget.snapshot()["bytes"] == 1000

    after = asyncio.create_task(budget.acquire(1, 1))
    await _settle()
    assert not after.done()
    budget.release(1, 1000)
    await asyncio.wait_for(after, 1)


@pytest.mark.asyncio
async def test_message_with_more_parts_than_limit_admitted_alone():
    budget = IntakeBudget(max_items=3, max_bytes=100)

    await asyncio.wait_for(budget.acquire(5, 10), 1)

    assert budget.snapshot()["items"] == 5


@pytest.mark.asyncio
async def test_parts_release_separately():
    budget = IntakeBudget(max_items=10, max_bytes=100)
    await budget.acquire(3, 90)

    budget.release(1, 30)
    budget.release(1, 30)

    assert budget.snapshot()["items"] == 1
    assert budget.snapshot()["bytes"] == 30


@pytest.mark.asyncio
async def test_cancelled_waiter_is_removed_and_next_admitted():
    budget = IntakeBudget(max_items=1, max_bytes=100)
    await budget.acquire(1, 10)
    first = asyncio.create_task(budget.acquire(1, 10))
    second = asyncio.create_task(budget.acquire(1, 10))
    await _settle()

    first.cancel()
    await asyncio.gather(first, return_exceptions=True)
    budget.release(1, 10)
    await asyncio.wait_for(second, 1)

    assert budget.snapshot() == {"items": 1, "max_items": 1, "bytes": 10, "max_bytes": 100, "waiting": 0}


@pytest.mark.parametrize(
    "items, size, expected",
    [(3, 100, False), (4, 100, True), (3, 101, True), (1, 1, False)],
)
def test_exceeds(items: int, size: int, expected: bool):
    budget = IntakeBudget(max_items=3, max_bytes=100)

    assert budget.exceeds(items, size) is expected
