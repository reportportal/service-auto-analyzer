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

"""Intake budget: bounds the number and the size of tasks the coordinator holds in memory."""

import asyncio
from collections import deque
from typing import Any


class IntakeBudget:
    """Item and byte budget for one consumer.

    A message acquires its share before it is acked and gives it back when its tasks finish. While a message waits,
    it stays unacked, so the broker stops delivering. Waiters are admitted strictly in arrival order.

    Used only from the event loop thread, so there are no locks.
    """

    _max_items: int
    _max_bytes: int
    _items: int
    _bytes: int
    _waiters: deque[tuple[int, int, asyncio.Future[None]]]

    def __init__(self, max_items: int, max_bytes: int) -> None:
        """Create the budget.

        :param max_items: Maximum number of queued and running tasks
        :param max_bytes: Maximum total body size of queued and running tasks
        """
        self._max_items = max_items
        self._max_bytes = max_bytes
        self._items = 0
        self._bytes = 0
        self._waiters = deque()

    @property
    def max_items(self) -> int:
        return self._max_items

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def exceeds(self, items: int, size: int) -> bool:
        """Whether a message is bigger than the whole budget, so it can only be admitted alone.

        :param items: Number of tasks of the message
        :param size: Total body size of the tasks of the message
        """
        return items > self._max_items or size > self._max_bytes

    def _fits(self, items: int, size: int) -> bool:
        if self._items == 0:
            # An oversized message is admitted alone rather than never, so it can't block intake forever
            return True
        return self._items + items <= self._max_items and self._bytes + size <= self._max_bytes

    def _take(self, items: int, size: int) -> None:
        self._items += items
        self._bytes += size

    async def acquire(self, items: int, size: int) -> None:
        """Wait until the budget has room for the message, then take its share.

        :param items: Number of tasks of the message
        :param size: Total body size of the tasks of the message
        """
        if not self._waiters and self._fits(items, size):
            self._take(items, size)
            return
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        waiter = (items, size, future)
        self._waiters.append(waiter)
        try:
            await future  # release() has already taken this message's share
        except asyncio.CancelledError:
            if future.done() and not future.cancelled():
                # The share was taken just before the cancellation: give it back
                self.release(items, size)
            elif waiter in self._waiters:
                # release() may already have dropped the canceled waiter
                self._waiters.remove(waiter)
                self._wake()
            raise

    def release(self, items: int, size: int) -> None:
        """Give back the share of finished tasks, and admit waiters which fit now.

        :param items: Number of finished tasks
        :param size: Total body size of the finished tasks
        """
        self._items -= items
        self._bytes -= size
        self._wake()

    def _wake(self) -> None:
        """Admit waiters from the head of the queue while they fit."""
        while self._waiters and self._fits(self._waiters[0][0], self._waiters[0][1]):
            items, size, future = self._waiters.popleft()
            if future.cancelled():
                continue
            self._take(items, size)
            future.set_result(None)

    def snapshot(self) -> dict[str, Any]:
        """State of the budget, for the health endpoint."""
        return {
            "items": self._items,
            "max_items": self._max_items,
            "bytes": self._bytes,
            "max_bytes": self._max_bytes,
            "waiting": len(self._waiters),
        }
