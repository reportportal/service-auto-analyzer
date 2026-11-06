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

"""Coordinators of the queues: the lane coordinator, which orders messages in per-project lanes and dispatches them
to workers, and the discarding coordinator for the queue of a disabled feature."""

import asyncio
import itertools
import time
from collections import deque
from typing import Any, Awaitable, Callable, Final, Optional, Protocol

from app.amqp.async_amqp import AsyncAmqpClient
from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.commons.model.processing import ProcessingItem, ProcessingResult
from app.coordinator import keys, messages
from app.coordinator.budget import IntakeBudget
from app.coordinator.lanes import Lanes
from app.coordinator.messages import IncomingMessage
from app.coordinator.pool import TaskPool, WorkerSlot

LOGGER = logging.getLogger("analyzerApp.coordinator")

WATCHDOG_INTERVAL: Final[float] = 0.5
SHUTDOWN_GRACE_PERIOD: Final[float] = 30.0

SplitFunc = Callable[[str, bytes], list[tuple[Optional[str], bytes]]]
RetryPredicate = Callable[[ProcessingItem, Optional[Exception]], bool]
ReplyFunc = Callable[[str, str, str], Awaitable[None]]


class Coordinator(Protocol):
    """A consumer of one queue, run on the event loop of the main process."""

    @property
    def name(self) -> str:
        """Name of the coordinator for logs and the health endpoint."""
        ...

    @property
    def connected(self) -> bool:
        """Whether the coordinator is connected to the broker."""
        ...

    async def start(self) -> None:
        """Connect and start consuming."""
        ...

    async def shutdown(self, grace_period: float = SHUTDOWN_GRACE_PERIOD) -> None:
        """Stop consuming and release the resources.

        :param grace_period: Seconds to wait for running tasks
        """
        ...

    async def on_message(self, message: IncomingMessage) -> None:
        """Take a message in, the only entry point of messages.

        :param message: The incoming message
        """
        ...

    def snapshot(self) -> dict[str, Any]:
        """State of the coordinator, for the health endpoint."""
        ...


class LaneCoordinator(Coordinator):
    """Processes the messages of one queue: tasks of one project in order, tasks of different projects in parallel.

    Everything here runs on the event loop thread, so the lanes, the budget and the workers need no locks.
    """

    _name: Final[str]
    _app_config: Final[ApplicationConfig]
    _queue: Final[str]
    _routing_key_predicate: Final[Callable[[str], bool]]
    _pool: Final[TaskPool]
    _budget: Final[IntakeBudget]
    _lanes: Final[Lanes]
    _client: Optional[AsyncAmqpClient]
    _reply: Optional[ReplyFunc]
    _split: Final[SplitFunc]
    _retry_predicate: Final[Optional[RetryPredicate]]
    _numbers: Final[Any]
    _retries: Final[deque[ProcessingItem]]
    _background: Final[set[asyncio.Task[Any]]]
    _watchdog: Optional[asyncio.Task[None]]
    _accepting: bool

    def __init__(
        self,
        app_config: ApplicationConfig,
        *,
        name: str,
        queue: str,
        routing_key_predicate: Callable[[str], bool],
        pool: TaskPool,
        budget: IntakeBudget,
        client: Optional[AsyncAmqpClient] = None,
        reply: Optional[ReplyFunc] = None,
        split_func: Optional[SplitFunc] = None,
        retry_predicate: Optional[RetryPredicate] = messages.retry,
    ) -> None:
        """Create the coordinator.

        :param app_config: Application configuration, for the task timeout and the number of retries
        :param name: Name for logs and the health endpoint
        :param queue: Name of the queue to consume
        :param routing_key_predicate: Which routing keys this coordinator processes; the exchange is a fanout, so every
            queue receives every message
        :param pool: Workers to run the tasks on
        :param budget: Intake budget of this coordinator
        :param client: AMQP client to consume with, and to reply with unless `reply` is given
        :param reply: Optional function which publishes replies, to replace the client's in tests
        :param split_func: Optional function which splits a message into per-project parts, to replace
            `keys.split_by_project` in tests
        :param retry_predicate: Whether a failed task should be retried
        """
        self._name = name
        self._app_config = app_config
        self._queue = queue
        self._routing_key_predicate = routing_key_predicate
        self._pool = pool
        self._budget = budget
        self._lanes = Lanes()
        self._client = client
        self._reply = reply or (client.reply if client else None)
        self._split = split_func or keys.split_by_project
        self._retry_predicate = retry_predicate
        self._numbers = itertools.count(1)
        self._retries = deque()
        self._background = set()
        self._watchdog = None
        self._accepting = True
        pool.bind(self._on_result, self._on_lost)

    @property
    def name(self) -> str:
        return self._name

    # Lifecycle

    async def start(self) -> None:
        """Start the workers and the watchdog, then connect and consume."""
        await self._pool.start()
        self._watchdog = asyncio.create_task(self._watch(), name=f"{self.name}-watchdog")
        if self._client is not None:
            await self._client.connect()
            await self._client.consume(
                self._queue, lambda message: self.on_message(IncomingMessage.from_aio_pika(message))
            )

    async def shutdown(self, grace_period: float = SHUTDOWN_GRACE_PERIOD) -> None:
        """Stop consuming, let running tasks finish within the grace period, then stop the workers.

        :param grace_period: Seconds to wait for running tasks
        """
        self._accepting = False
        if self._client is not None:
            await self._client.stop_consuming()
        deadline = time.monotonic() + grace_period
        while self._running_items() and time.monotonic() < deadline:
            await asyncio.sleep(0.1)
        dropped = self._lanes.drain() + list(self._retries) + self._running_items()
        self._retries.clear()
        for item in dropped:
            LOGGER.warning(
                f"Dropping a task at shutdown. --Routing key: {item.routing_key} "
                f"--Correlation ID: {item.msg_correlation_id} --Lane: {item.lane}",
                correlation_id=item.log_correlation_id,
            )
        if self._watchdog is not None:
            self._watchdog.cancel()
        await self._pool.shutdown()
        if self._background:
            await asyncio.gather(*self._background, return_exceptions=True)
        if self._client is not None:
            await self._client.close()
        LOGGER.info(f"Coordinator '{self.name}' stopped, {len(dropped)} tasks dropped")

    def _spawn(self, coroutine: Awaitable[Any]) -> None:
        task = asyncio.ensure_future(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _running_items(self) -> list[ProcessingItem]:
        return [slot.item for slot in self._pool.slots if slot.item is not None]

    # Intake

    async def on_message(self, message: IncomingMessage) -> None:
        """Take a message in: parse, split by project, wait for the budget, ack, and queue its tasks.

        :param message: The incoming message
        """
        # The arrival number is assigned before any await, so it follows delivery order
        number = next(self._numbers)
        correlation_id = logging.new_correlation_id()
        routing_key = message.routing_key
        if not self._routing_key_predicate(routing_key):
            await message.ack()
            return
        try:
            parts = self._split(routing_key, message.body)
        except Exception as exc:
            LOGGER.exception("Failed to parse message body to JSON", exc_info=exc)
            await message.reject()
            return

        priority = messages.get_priority(message.headers)
        items = [
            ProcessingItem(
                priority=priority,
                number=number if index == 0 else next(self._numbers),
                routing_key=routing_key,
                reply_to=message.reply_to,
                log_correlation_id=correlation_id,
                msg_correlation_id=message.correlation_id or "",
                item=raw,
                lane=lane,
            )
            for index, (lane, raw) in enumerate(parts)
        ]
        size = sum(item.size for item in items)
        if self._budget.exceeds(len(items), size):
            LOGGER.warning(
                f"Message {routing_key} - {message.correlation_id} of {len(items)} tasks and {size} bytes exceeds the "
                f"intake budget of '{self.name}' ({self._budget.max_items} tasks, {self._budget.max_bytes} bytes). "
                "It waits until all held tasks finish and pauses intake until it finishes itself"
            )
        await self._budget.acquire(len(items), size)
        if not self._accepting:
            self._budget.release(len(items), size)
            return
        await message.ack()
        for item in items:
            self._lanes.push(item)
        self._dispatch()

    # Dispatch

    def _dispatch(self) -> None:
        """Choose the tasks to run while the pool has idle workers: retries first, then the oldest task of an idle
        lane. The pool chooses the worker."""
        while self._pool.has_idle():
            item = self._retries.popleft() if self._retries else self._lanes.pop_ready()
            if item is None:
                return
            logging.set_correlation_id(item.log_correlation_id)
            messages.log_incoming_message(item)
            self._pool.submit(item)

    def _finish(self, item: ProcessingItem) -> None:
        """Free the lane and the budget share of a task which won't run again."""
        self._lanes.release(item.lane)
        self._budget.release(1, item.size)

    def _retry_later(self, item: ProcessingItem) -> None:
        """Queue a task for another attempt. Its lane stays busy, so nothing in the project overtakes it."""
        self._retries.append(item)

    # Results

    def _on_result(self, slot: WorkerSlot, result: ProcessingResult) -> None:
        item = slot.item
        slot.item = None
        if item is None or item.number != result.number:
            LOGGER.warning(f"Worker {slot.index} returned a result for an unexpected task {result.number}")
            self._dispatch()
            return
        logging.set_correlation_id(item.log_correlation_id)
        if result.success:
            if item.retries > 0:
                LOGGER.warning(f"Task {item.routing_key} - {item.msg_correlation_id} was retried {item.retries} times")
            self._finish(item)
            if result.result is not None and item.reply_to:
                self._spawn(self._publish_reply(item, result.result))
        else:
            self._handle_failure(item, result.error)
        self._dispatch()

    def _handle_failure(self, item: ProcessingItem, error: Optional[Exception]) -> None:
        should_retry = bool(error)
        if self._retry_predicate is not None and error is not None:
            should_retry = self._retry_predicate(item, error)
            if not should_retry:
                LOGGER.info(
                    f"Retry predicate failed for message {item.routing_key} - {item.msg_correlation_id}, not retrying"
                )
        if not should_retry or item.retries >= self._app_config.amqpHandlerMaxRetries:
            LOGGER.error(
                f"Task failed after {item.retries} retries. --Routing key: {item.routing_key}"
                f" --Correlation ID: {item.msg_correlation_id} --Lane: {item.lane}"
            )
            self._finish(item)
            return
        item.retries += 1
        LOGGER.debug(
            f"Retrying task {item.routing_key} - {item.msg_correlation_id}, "
            f"attempt {item.retries} of {self._app_config.amqpHandlerMaxRetries}"
        )
        self._retry_later(item)

    async def _publish_reply(self, item: ProcessingItem, response_body: Any) -> None:
        if self._reply is None or not item.reply_to:
            return
        logging.set_correlation_id(item.log_correlation_id)
        try:
            messages.log_outgoing_message(item.reply_to, item.msg_correlation_id, response_body)
            await self._reply(item.reply_to, item.msg_correlation_id, response_body)
        except Exception as exc:
            LOGGER.exception("Failed to publish result", exc_info=exc)

    # Hung and dead workers

    def _on_lost(self, slot: WorkerSlot) -> None:
        """A worker died, timed out or couldn't take a task: retry its task elsewhere and restart it."""
        if slot.restarting:
            return
        item = slot.item
        slot.item = None
        slot.restarting = True  # Not idle until restarted
        if item is not None:
            logging.set_correlation_id(item.log_correlation_id)
            item.retries += 1
            if item.retries >= self._app_config.amqpHandlerMaxRetries:
                LOGGER.warning(
                    f"Task exceeded max retries, skipping. --Routing key: {item.routing_key}"
                    f" --Correlation ID: {item.msg_correlation_id} --Lane: {item.lane}",
                    correlation_id=item.log_correlation_id,
                )
                self._finish(item)
            else:
                self._retry_later(item)
        self._spawn(self._restart(slot))
        self._dispatch()

    async def _restart(self, slot: WorkerSlot) -> None:
        try:
            await self._pool.restart(slot)
            LOGGER.info(f"Successfully restarted worker {slot.index} of '{self.name}'")
        except Exception as exc:
            LOGGER.exception(f"Failed to restart worker {slot.index} of '{self.name}'", exc_info=exc)
            slot.restarting = False
            return
        self._dispatch()

    async def _watch(self) -> None:
        """Restart workers which run a task longer than the timeout, or which died."""
        timeout = self._app_config.amqpHandlerTaskTimeout
        while True:
            await asyncio.sleep(WATCHDOG_INTERVAL)
            now = time.monotonic()
            for slot in self._pool.slots:
                if slot.restarting:
                    continue
                if slot.item is not None and now - slot.started > timeout and self._pool.can_kill:
                    LOGGER.warning(
                        f"Task {slot.item.routing_key} - {slot.item.msg_correlation_id} is taking too long, "
                        f"restarting worker {slot.index}",
                        correlation_id=slot.item.log_correlation_id,
                    )
                    self._on_lost(slot)
                elif slot.processor is not None and not slot.processor.is_alive():
                    LOGGER.warning(f"Worker {slot.index} of '{self.name}' is not running, restarting")
                    self._on_lost(slot)

    # Health

    @property
    def connected(self) -> bool:
        """Whether the consumer is connected, or True if this coordinator doesn't consume (tests)."""
        return self._client is None or self._client.connected

    def snapshot(self) -> dict[str, Any]:
        """State of the coordinator, for the health endpoint."""
        return {
            "name": self.name,
            "connected": self.connected,
            "workers": [slot.snapshot() for slot in self._pool.slots],
            "retrying": len(self._retries),
            **self._lanes.snapshot(),
            "budget": self._budget.snapshot(),
        }


class DiscardingCoordinator(Coordinator):
    """Acks and drops every message of a queue, so the queue of a disabled feature doesn't grow forever.

    The analyzer exchange is a fanout one, so the queue gets every message published to the analyzer, not only the
    ones of the disabled feature.
    """

    _name: Final[str]
    _queue: Final[str]
    _client: Optional[AsyncAmqpClient]
    _discarded: int

    def __init__(self, *, name: str, queue: str, client: Optional[AsyncAmqpClient] = None) -> None:
        """Create the coordinator, it connects on `start`.

        :param name: Name for logs and the health endpoint
        :param queue: Name of the queue to consume
        :param client: AMQP client to consume with, or None to only take messages passed to `on_message` (tests)
        """
        self._name = name
        self._queue = queue
        self._client = client
        self._discarded = 0

    @property
    def name(self) -> str:
        return self._name

    # Lifecycle

    async def start(self) -> None:
        """Connect and start discarding the messages of the queue."""
        LOGGER.info(f"Coordinator '{self.name}' discards all messages of queue '{self._queue}'")
        if self._client is not None:
            await self._client.connect()
            await self._client.consume(
                self._queue, lambda message: self.on_message(IncomingMessage.from_aio_pika(message))
            )

    async def shutdown(self, grace_period: float = SHUTDOWN_GRACE_PERIOD) -> None:
        """Stop consuming and close the connection. Nothing runs, so there is nothing to wait for.

        :param grace_period: Unused, for compatibility with the other coordinators
        """
        if self._client is not None:
            await self._client.stop_consuming()
            await self._client.close()
        LOGGER.info(f"Coordinator '{self.name}' stopped, {self._discarded} messages discarded")

    # Intake

    async def on_message(self, message: IncomingMessage) -> None:
        """Ack a message and drop it.

        :param message: The incoming message
        """
        await message.ack()
        self._discarded += 1
        LOGGER.debug(f"Discarded a '{message.routing_key}' message of queue '{self._queue}'")

    # Health

    @property
    def connected(self) -> bool:
        """Whether the consumer is connected, or True if this coordinator doesn't consume (tests)."""
        return self._client is None or self._client.connected

    def snapshot(self) -> dict[str, Any]:
        """State of the coordinator, for the health endpoint."""
        return {"name": self.name, "connected": self.connected, "discarding": True, "discarded": self._discarded}
