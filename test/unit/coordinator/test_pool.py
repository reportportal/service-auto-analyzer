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

"""Worker selection of the pool, with mocked worker processes."""

from unittest import mock

import pytest

from app.commons.model.processing import ProcessingItem
from app.coordinator.pool import WorkerPool


def item(number: int) -> ProcessingItem:
    return ProcessingItem(
        priority=0,
        number=number,
        routing_key="suggest",
        reply_to=None,
        log_correlation_id="log",
        msg_correlation_id="msg",
        item=b"{}",
        lane="1",
    )


def started_pool(size: int) -> tuple[WorkerPool, mock.Mock]:
    pool = WorkerPool(mock.Mock(), size, "test")
    for slot in pool.slots:
        slot.processor = mock.Mock()
    on_lost = mock.Mock()
    pool.bind(mock.Mock(), on_lost)
    return pool, on_lost


def test_submit_sends_to_the_first_idle_worker():
    pool, _ = started_pool(3)
    busy, restarting, idle = pool.slots
    busy.item = item(1)
    restarting.restarting = True

    task = item(2)
    pool.submit(task)

    idle.processor.send.assert_called_once_with(task)
    busy.processor.send.assert_not_called()
    restarting.processor.send.assert_not_called()
    assert idle.item is task
    assert task.send_time == idle.started


def test_no_idle_worker():
    pool, _ = started_pool(2)
    pool.submit(item(1))
    pool.submit(item(2))

    assert pool.has_idle() is False
    with pytest.raises(RuntimeError):
        pool.submit(item(3))


def test_workers_which_are_not_started_are_not_idle():
    pool = WorkerPool(mock.Mock(), 2, "test")

    assert pool.has_idle() is False
    with pytest.raises(RuntimeError):
        pool.submit(item(1))


def test_send_failure_reports_the_worker_lost_with_its_task():
    pool, on_lost = started_pool(1)
    slot = pool.slots[0]
    slot.processor.send.side_effect = BrokenPipeError()

    task = item(1)
    pool.submit(task)

    on_lost.assert_called_once_with(slot)
    assert slot.item is task  # The coordinator takes the task from the slot to retry it
