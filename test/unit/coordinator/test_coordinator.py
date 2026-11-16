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

"""Coordinator tests with real worker processes, which run the "noop_*" test services."""

import asyncio
import json
import pickle
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Awaitable, Callable, Optional
from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem
from app.commons.processing import Worker
from app.coordinator import messages
from app.coordinator.budget import IntakeBudget
from app.coordinator.coordinator import LaneCoordinator, SplitFunc
from app.coordinator.messages import IncomingMessage
from app.coordinator.pool import WorkerPool

NOOP_SERVICES = {"noop_echo", "noop_sleep", "noop_fail"}
WAIT_TIMEOUT = 30.0


@dataclass
class Harness:
    coordinator: LaneCoordinator
    replies: list[tuple[str, float]] = field(default_factory=list)
    failures: list[float] = field(default_factory=list)

    def reply_values(self) -> list[str]:
        return [value for value, _ in self.replies]

    def reply_time(self, value: str) -> float:
        return next(at for reply_value, at in self.replies if reply_value == value)

    def pids(self) -> list[Optional[int]]:
        return [worker["pid"] for worker in self.coordinator.snapshot()["workers"]]

    def held_items(self) -> int:
        return self.coordinator.snapshot()["budget"]["items"]


def lane_from_body(routing_key: str, raw: bytes) -> list[tuple[Optional[str], bytes]]:
    """Test lanes: "noop_echo" and "noop_fail" bodies are "<lane>:<tag>", "noop_sleep" bodies are [seconds, lane]."""
    body = json.loads(raw)
    if routing_key == "noop_sleep" and isinstance(body, list):
        return [(body[1], json.dumps(body[0]).encode("utf-8"))]
    if isinstance(body, str) and ":" in body:
        return [(body.split(":")[0], raw)]
    return [(None, raw)]


async def wait_until(condition: Callable[[], bool], timeout: float = WAIT_TIMEOUT) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("Condition was not met in time")
        await asyncio.sleep(0.02)


def raw_message(routing_key: str, body: bytes, correlation_id: str = "corr") -> IncomingMessage:
    return IncomingMessage(
        routing_key=routing_key,
        body=body,
        headers=None,
        reply_to="reply-queue",
        correlation_id=correlation_id,
        ack=AsyncMock(),
        reject=AsyncMock(),
    )


def message(routing_key: str, body: Any, correlation_id: str = "corr") -> IncomingMessage:
    return raw_message(routing_key, json.dumps(body).encode("utf-8"), correlation_id)


async def send(harness: Harness, routing_key: str, body: Any) -> tuple[AsyncMock, AsyncMock]:
    incoming = message(routing_key, body)
    await harness.coordinator.on_message(incoming)
    return incoming.ack, incoming.reject


@pytest.fixture
def app_config(tmp_path) -> ApplicationConfig:
    return ApplicationConfig(
        logLevel="INFO",
        analyzerPathToLog=str(tmp_path / "test.log"),
        appVersion="test",
        amqpHandlerMaxRetries=2,
        amqpHandlerTaskTimeout=3,
    )


HarnessFactory = Callable[..., Awaitable[Harness]]


@pytest_asyncio.fixture
async def make_harness(app_config) -> AsyncGenerator[HarnessFactory, None]:
    created: list[Harness] = []

    async def factory(
        *,
        workers: int = 2,
        max_items: int = 50,
        split_func: SplitFunc = lane_from_body,
    ) -> Harness:
        harness: Optional[Harness] = None

        async def reply(_: str, __: str, data: str) -> None:
            assert harness is not None
            harness.replies.append((data, time.monotonic()))

        def recording_retry(item: ProcessingItem, exc: Optional[Exception]) -> bool:
            assert harness is not None
            harness.failures.append(time.monotonic())
            return messages.retry(item, exc)

        coordinator = LaneCoordinator(
            app_config,
            name="test",
            queue="all",
            routing_key_predicate=lambda routing_key: routing_key != "train_models",
            pool=WorkerPool.of_workers(app_config, SearchConfig(), NOOP_SERVICES, workers),
            budget=IntakeBudget(max_items, 10_000_000),
            reply=reply,
            split_func=split_func,
            retry_predicate=recording_retry,
        )
        harness = Harness(coordinator)
        await coordinator.start()
        created.append(harness)
        # Warm up: every worker imports the services on its first task
        for index in range(workers):
            await send(harness, "noop_sleep", [0.5, f"warm-up-{index}"])
        await wait_until(lambda: harness.held_items() == 0)
        return harness

    yield factory
    for harness in created:
        await harness.coordinator.shutdown(grace_period=0)


def test_worker_target_is_picklable(app_config):
    target = Worker(app_config, SearchConfig(), NOOP_SERVICES).work

    restored = pickle.loads(pickle.dumps(target))

    assert restored.__self__.app_config == app_config


@pytest.mark.asyncio
async def test_different_projects_run_in_parallel(make_harness):
    harness = await make_harness()
    start = time.monotonic()

    await send(harness, "noop_sleep", [3.0, "A"])
    await send(harness, "noop_echo", "B:1")
    await wait_until(lambda: harness.reply_values() == ["B:1"])

    assert harness.reply_time("B:1") - start < 2.0
    running = [worker["task"]["lane"] for worker in harness.coordinator.snapshot()["workers"] if "task" in worker]
    assert running == ["A"]


@pytest.mark.asyncio
async def test_one_project_runs_in_order(make_harness):
    harness = await make_harness()
    start = time.monotonic()

    await send(harness, "noop_sleep", [1.5, "A"])
    for value in ("A:1", "A:2", "A:3", "B:1"):
        await send(harness, "noop_echo", value)
    await wait_until(lambda: len(harness.replies) == 4)

    assert harness.reply_values() == ["B:1", "A:1", "A:2", "A:3"]
    assert harness.reply_time("B:1") - start < 1.5
    assert harness.reply_time("A:1") - start >= 1.5


@pytest.mark.asyncio
async def test_retry_holds_the_lane(make_harness):
    harness = await make_harness()

    await send(harness, "noop_fail", "A:fail")
    await send(harness, "noop_echo", "A:1")
    await wait_until(lambda: harness.reply_values() == ["A:1"])

    # The first attempt and 2 retries, all before the next task of the lane
    assert len(harness.failures) == 3
    assert max(harness.failures) < harness.reply_time("A:1")
    assert harness.held_items() == 0


@pytest.mark.asyncio
async def test_watchdog_restarts_only_the_hung_worker(make_harness):
    harness = await make_harness()
    initial_pids = harness.pids()

    await send(harness, "noop_sleep", [30.0, "A"])
    await send(harness, "noop_echo", "B:1")
    await wait_until(lambda: harness.reply_values() == ["B:1"])
    await wait_until(lambda: harness.pids() != initial_pids and None not in harness.pids())

    changed = [before != after for before, after in zip(initial_pids, harness.pids())]
    assert changed.count(True) == 1
    # The hung task is retried until it exceeds the retries, then dropped
    await wait_until(lambda: harness.held_items() == 0)
    assert harness.reply_values() == ["B:1"]


@pytest.mark.asyncio
async def test_budget_pauses_intake(make_harness):
    harness = await make_harness(max_items=1)

    first_ack, _ = await send(harness, "noop_sleep", [1.5, "A"])
    first_ack.assert_awaited_once()
    second_message = message("noop_echo", "B:1")
    second_ack = second_message.ack
    second = asyncio.create_task(harness.coordinator.on_message(second_message))
    await asyncio.sleep(0.5)

    second_ack.assert_not_awaited()
    assert harness.coordinator.snapshot()["budget"]["waiting"] == 1

    await asyncio.wait_for(second, WAIT_TIMEOUT)
    second_ack.assert_awaited_once()
    await wait_until(lambda: harness.reply_values() == ["B:1"])


@pytest.mark.asyncio
async def test_multi_project_parts_run_in_parallel(make_harness):
    def two_projects(routing_key: str, raw: bytes) -> list[tuple[Optional[str], bytes]]:
        if routing_key == "noop_sleep" and json.loads(raw) == 1.5:
            return [("A", raw), ("B", raw)]
        return lane_from_body(routing_key, raw)

    harness = await make_harness(split_func=two_projects)
    start = time.monotonic()

    ack, _ = await send(harness, "noop_sleep", 1.5)
    assert harness.held_items() == 2
    await wait_until(lambda: harness.held_items() == 0)

    ack.assert_awaited_once()
    assert time.monotonic() - start < 2.8


@pytest.mark.asyncio
async def test_malformed_json_is_rejected(make_harness):
    harness = await make_harness(workers=1)
    incoming = raw_message("noop_echo", b"{not json")

    await harness.coordinator.on_message(incoming)

    incoming.reject.assert_awaited_once()
    incoming.ack.assert_not_awaited()
    assert harness.held_items() == 0


@pytest.mark.asyncio
async def test_foreign_routing_key_is_dropped_before_the_budget(make_harness):
    harness = await make_harness(workers=1)

    ack, reject = await send(harness, "train_models", {"project": 1})

    ack.assert_awaited_once()
    reject.assert_not_awaited()
    assert harness.held_items() == 0
    assert all("task" not in worker for worker in harness.coordinator.snapshot()["workers"])
