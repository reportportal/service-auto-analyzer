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

"""Coordinator retry accounting with a fake pool, which lets the test decide every result."""

import json
import logging
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from app.commons.model.launch_objects import ApplicationConfig
from app.commons.model.processing import ProcessingItem, ProcessingResult, Reply
from app.coordinator.budget import IntakeBudget
from app.coordinator.coordinator import LaneCoordinator
from app.coordinator.messages import IncomingMessage
from app.coordinator.pool import LostCallback, ResultCallback, TaskPool, WorkerSlot


class FakePool(TaskPool):
    def __init__(self) -> None:
        self._slots = [WorkerSlot(0)]
        self.submitted: list[ProcessingItem] = []
        self.on_result: Optional[ResultCallback] = None
        self.on_lost: Optional[LostCallback] = None

    @property
    def slots(self) -> list[WorkerSlot]:
        return self._slots

    @property
    def can_kill(self) -> bool:
        return True

    def bind(self, on_result: ResultCallback, on_lost: LostCallback) -> None:
        self.on_result = on_result
        self.on_lost = on_lost

    async def start(self) -> None:
        pass

    def has_idle(self) -> bool:
        return self._slots[0].idle

    def submit(self, item: ProcessingItem) -> None:
        slot = self._slots[0]
        assert slot.idle, "the coordinator must submit only while the pool has an idle worker"
        slot.item = item
        self.submitted.append(item)

    async def restart(self, slot: WorkerSlot) -> None:
        slot.restarting = False

    async def shutdown(self) -> None:
        pass

    def finish(self, success: bool, replies: Optional[list[Reply]] = None) -> None:
        slot = self._slots[0]
        assert slot.item is not None
        assert self.on_result is not None
        error = None if success else RuntimeError("failure")
        self.on_result(slot, ProcessingResult(slot.item.number, replies, success, error))


def make_coordinator(pool: FakePool, budget: IntakeBudget) -> LaneCoordinator:
    return LaneCoordinator(
        ApplicationConfig(amqpHandlerMaxRetries=2),
        name="test",
        queue="all",
        routing_key_predicate=lambda _: True,
        pool=pool,
        budget=budget,
        reply=AsyncMock(),
    )


def suggest_message(project: int) -> IncomingMessage:
    body = json.dumps({"project": project}).encode("utf-8")
    return IncomingMessage("suggest", body, None, None, "corr", ack=AsyncMock(), reject=AsyncMock())


@pytest.mark.asyncio
async def test_budget_and_lane_are_held_through_retries():
    pool, budget = FakePool(), IntakeBudget(10, 1000)
    coordinator = make_coordinator(pool, budget)
    await coordinator.on_message(suggest_message(1))
    await coordinator.on_message(suggest_message(1))
    first_number = pool.submitted[0].number

    pool.finish(success=False)
    pool.finish(success=False)

    # Both failures were retried with the same task, the second task of the lane is still waiting
    assert [item.number for item in pool.submitted] == [first_number] * 3
    assert pool.submitted[0].retries == 2
    assert budget.snapshot()["items"] == 2

    pool.finish(success=False)

    # Max retries exceeded: the share is released and the next task of the lane runs
    assert len(pool.submitted) == 4
    assert pool.submitted[3].number != first_number
    assert budget.snapshot()["items"] == 1

    pool.finish(success=True)
    assert budget.snapshot()["items"] == 0


@pytest.mark.asyncio
async def test_lost_worker_charges_only_its_task():
    pool, budget = FakePool(), IntakeBudget(10, 1000)
    coordinator = make_coordinator(pool, budget)
    await coordinator.on_message(suggest_message(1))
    slot = pool.slots[0]
    assert slot.item is not None
    assert pool.on_lost is not None

    pool.on_lost(slot)

    assert pool.submitted[0].retries == 1
    assert budget.snapshot()["items"] == 1
    await coordinator.shutdown(grace_period=0)


@pytest.mark.asyncio
async def test_oversized_message_logs_warning(caplog: pytest.LogCaptureFixture):
    pool, budget = FakePool(), IntakeBudget(10, 5)
    coordinator = make_coordinator(pool, budget)
    message = suggest_message(1)

    with caplog.at_level(logging.WARNING, logger="analyzerApp.coordinator"):
        await coordinator.on_message(message)

    assert caplog.messages == [
        f"Message suggest - corr of 1 tasks and {len(message.body)} bytes exceeds the intake budget of 'test' "
        "(10 tasks, 5 bytes). It waits until all held tasks finish and pauses intake until it finishes itself"
    ]
    assert budget.snapshot()["items"] == 1
    await coordinator.shutdown(grace_period=0)


@pytest.mark.asyncio
async def test_message_within_budget_logs_nothing(caplog: pytest.LogCaptureFixture):
    pool, budget = FakePool(), IntakeBudget(10, 1000)
    coordinator = make_coordinator(pool, budget)

    with caplog.at_level(logging.WARNING, logger="analyzerApp.coordinator"):
        await coordinator.on_message(suggest_message(1))

    assert caplog.messages == []
    await coordinator.shutdown(grace_period=0)
