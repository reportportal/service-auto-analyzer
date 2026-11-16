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

"""Pools of workers which run tasks for the coordinator, one task per worker at a time."""

import asyncio
import time
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional, Protocol

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem, ProcessingResult
from app.commons.processing import RealProcessor, Worker

LOGGER = logging.getLogger("analyzerApp.coordinator.pool")


@dataclass
class WorkerSlot:
    """One worker and the task it runs."""

    index: int
    processor: Optional[RealProcessor] = None
    item: Optional[ProcessingItem] = None
    started: float = 0.0
    restarting: bool = False
    reading: bool = False

    @property
    def idle(self) -> bool:
        return self.item is None and not self.restarting

    def snapshot(self) -> dict[str, Any]:
        """State of the worker, for the health endpoint."""
        status: dict[str, Any] = {
            "pid": self.processor.pid if self.processor else None,
            "alive": self.processor.is_alive() if self.processor else not self.restarting,
            "restarting": self.restarting,
        }
        if self.item is not None:
            status["task"] = {
                "routing_key": self.item.routing_key,
                "correlation_id": self.item.msg_correlation_id,
                "lane": self.item.lane,
                "running_for": round(time.monotonic() - self.started, 3),
                "retries": self.item.retries,
            }
        return status


ResultCallback = Callable[[WorkerSlot, ProcessingResult], None]
LostCallback = Callable[[WorkerSlot], None]


class TaskPool(Protocol):
    """Workers the coordinator dispatches tasks to."""

    @property
    def slots(self) -> list[WorkerSlot]:
        """Workers of the pool."""
        ...

    @property
    def can_kill(self) -> bool:
        """Whether a running task can be stopped by restarting its worker."""
        ...

    def bind(self, on_result: ResultCallback, on_lost: LostCallback) -> None:
        """Set the callbacks for task results and for lost workers."""
        ...

    async def start(self) -> None:
        """Start the workers."""
        ...

    def has_idle(self) -> bool:
        """Whether a worker can take a task now."""
        ...

    def submit(self, item: ProcessingItem) -> None:
        """Send a task to an idle worker, chosen by the pool.

        :param item: The task
        :raises RuntimeError: If no worker is idle
        """
        ...

    async def restart(self, slot: WorkerSlot) -> None:
        """Stop a worker and start a new one in its place."""
        ...

    async def shutdown(self) -> None:
        """Stop all workers."""
        ...


def _ignore_result(_: WorkerSlot, __: ProcessingResult) -> None:
    LOGGER.warning("Received a result before the pool was bound to a coordinator")


def _ignore_lost(_: WorkerSlot) -> None:
    LOGGER.warning("Lost a worker before the pool was bound to a coordinator")


class WorkerPool(TaskPool):
    """Worker processes, each with a pipe which the event loop watches for results."""

    _target: Callable[[Connection], None]
    _name: str
    _slots: list[WorkerSlot]
    _on_result: ResultCallback
    _on_lost: LostCallback

    def __init__(self, target: Callable[[Connection], None], size: int, name: str = "pool") -> None:
        """Create the pool, the processes start on `start`.

        :param target: Picklable function to run in every worker process
        :param size: Number of worker processes
        :param name: Name of the pool for logs
        """
        self._target = target
        self._name = name
        self._slots = [WorkerSlot(index) for index in range(max(1, size))]
        self._on_result = _ignore_result
        self._on_lost = _ignore_lost

    @classmethod
    def of_workers(
        cls,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        init_services: set[str],
        size: int,
        name: str = "pool",
    ) -> "WorkerPool":
        """Create a pool of processes which run `ServiceProcessor` with the given services."""
        return cls(Worker(app_config, search_config, init_services).work, size, name)

    @property
    def slots(self) -> list[WorkerSlot]:
        return self._slots

    @property
    def can_kill(self) -> bool:
        return True

    def bind(self, on_result: ResultCallback, on_lost: LostCallback) -> None:
        """Set the callbacks for results and for workers which died.

        :param on_result: Called with the slot and the result of its task
        :param on_lost: Called with the slot of a worker which died
        """
        self._on_result = on_result
        self._on_lost = on_lost

    def _attach(self, slot: WorkerSlot) -> None:
        if slot.processor is not None and not slot.reading:
            asyncio.get_running_loop().add_reader(slot.processor.fileno(), self._on_readable, slot)
            slot.reading = True

    def _detach(self, slot: WorkerSlot) -> None:
        if slot.processor is not None and slot.reading:
            try:
                asyncio.get_running_loop().remove_reader(slot.processor.fileno())
            except (OSError, ValueError):
                pass
            slot.reading = False

    def _on_readable(self, slot: WorkerSlot) -> None:
        processor = slot.processor
        if processor is None:
            return
        try:
            if not processor.poll():
                return
            result = processor.recv()
        except (EOFError, OSError):
            # The worker process died: its end of the pipe is closed
            self._detach(slot)
            self._on_lost(slot)
            return
        self._on_result(slot, result)

    async def _start_slot(self, slot: WorkerSlot) -> None:
        slot.processor = await asyncio.to_thread(RealProcessor, self._target)
        self._attach(slot)
        LOGGER.info(f"Worker {slot.index} of '{self._name}' started, PID {slot.processor.pid}")

    async def start(self) -> None:
        """Start all worker processes."""
        await asyncio.gather(*(self._start_slot(slot) for slot in self._slots))

    def _idle_slot(self) -> Optional[WorkerSlot]:
        return next((slot for slot in self._slots if slot.idle and slot.processor is not None), None)

    def has_idle(self) -> bool:
        """Whether a started worker is free to take a task."""
        return self._idle_slot() is not None

    def submit(self, item: ProcessingItem) -> None:
        """Send a task to the first idle worker.

        A worker whose pipe can't take the task is dead: it is reported to the coordinator as lost, which retries the
        task and restarts the worker.

        :param item: The task
        :raises RuntimeError: If no worker is idle
        """
        slot = self._idle_slot()
        if slot is None or slot.processor is None:
            raise RuntimeError(f"No idle worker in '{self._name}'")
        slot.item = item
        slot.started = time.monotonic()
        item.send_time = slot.started
        try:
            slot.processor.send(item)
        except Exception as exc:
            LOGGER.exception(f"Failed to send a task to worker {slot.index} of '{self._name}'", exc_info=exc)
            self._detach(slot)
            self._on_lost(slot)

    async def restart(self, slot: WorkerSlot) -> None:
        """Kill a worker process and start a new one in its place. The task it ran is the caller's to handle.

        :param slot: The worker to restart
        """
        slot.restarting = True
        try:
            self._detach(slot)
            if slot.processor is not None:
                await asyncio.to_thread(slot.processor.kill)
            slot.item = None
            await self._start_slot(slot)
        finally:
            slot.restarting = False

    async def shutdown(self) -> None:
        """Stop all worker processes."""

        async def stop(slot: WorkerSlot) -> None:
            self._detach(slot)
            if slot.processor is not None:
                await asyncio.to_thread(slot.processor.shutdown)

        await asyncio.gather(*(stop(slot) for slot in self._slots))


class InlinePool(TaskPool):
    """One worker thread in the coordinator process, for debugging. A running task can't be stopped."""

    _app_config: ApplicationConfig
    _search_config: SearchConfig
    _init_services: set[str]
    _slots: list[WorkerSlot]
    _processor: Any
    _on_result: ResultCallback
    _tasks: set[asyncio.Task[None]]

    def __init__(self, app_config: ApplicationConfig, search_config: SearchConfig, init_services: set[str]) -> None:
        self._app_config = app_config
        self._search_config = search_config
        self._init_services = init_services
        self._slots = [WorkerSlot(0)]
        self._processor = None
        self._on_result = _ignore_result
        self._tasks = set()

    @property
    def slots(self) -> list[WorkerSlot]:
        return self._slots

    @property
    def can_kill(self) -> bool:
        return False

    def bind(self, on_result: ResultCallback, on_lost: LostCallback) -> None:
        self._on_result = on_result

    async def start(self) -> None:
        # Imported here: the services pull in the ML libraries, which only the debug mode loads into this process
        from app.service.processor import ServiceProcessor

        self._processor = await asyncio.to_thread(
            ServiceProcessor, self._app_config, self._search_config, services_to_init=self._init_services
        )

    async def _run(self, slot: WorkerSlot, item: ProcessingItem) -> None:
        result = await asyncio.to_thread(Worker.process, item, self._processor)
        self._on_result(slot, result)

    def has_idle(self) -> bool:
        return self._processor is not None and self._slots[0].idle

    def submit(self, item: ProcessingItem) -> None:
        slot = self._slots[0]
        if not self.has_idle():
            raise RuntimeError("The debug worker is busy")
        slot.item = item
        slot.started = time.monotonic()
        item.send_time = slot.started
        task = asyncio.create_task(self._run(slot, item))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def restart(self, slot: WorkerSlot) -> None:
        LOGGER.warning("A task in debug mode can't be stopped, it keeps running")

    async def shutdown(self) -> None:
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
