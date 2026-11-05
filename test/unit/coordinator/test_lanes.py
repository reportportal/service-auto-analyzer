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

from typing import Optional

from app.commons.model.processing import ProcessingItem
from app.coordinator.lanes import Lanes


def _item(number: int, lane: Optional[str], priority: int = 1000) -> ProcessingItem:
    return ProcessingItem(
        priority=priority,
        number=number,
        routing_key="noop_echo",
        reply_to=None,
        log_correlation_id=f"log-{number}",
        msg_correlation_id=f"msg-{number}",
        item=b"1",
        lane=lane,
    )


def test_lane_runs_one_task_at_a_time_in_order():
    lanes = Lanes()
    for number in (1, 2, 3):
        lanes.push(_item(number, "A"))

    first = lanes.pop_ready()
    assert first is not None and first.number == 1
    assert lanes.pop_ready() is None

    lanes.release("A")
    second = lanes.pop_ready()
    assert second is not None and second.number == 2


def test_different_lanes_run_in_parallel_oldest_head_first():
    lanes = Lanes()
    lanes.push(_item(3, "B", priority=30))
    lanes.push(_item(1, "A", priority=10))
    lanes.push(_item(2, "A", priority=20))

    taken = [lanes.pop_ready(), lanes.pop_ready(), lanes.pop_ready()]

    assert [item.number if item else None for item in taken] == [1, 3, None]


def test_priority_orders_within_lane():
    lanes = Lanes()
    lanes.push(_item(1, "A", priority=200))
    lanes.push(_item(2, "A", priority=100))

    first = lanes.pop_ready()

    assert first is not None and first.number == 2


def test_free_tasks_do_not_wait_for_each_other():
    lanes = Lanes()
    lanes.push(_item(1, None))
    lanes.push(_item(2, None))

    taken = [lanes.pop_ready(), lanes.pop_ready()]

    assert [item.number if item else None for item in taken] == [1, 2]


def test_empty_idle_lane_is_forgotten():
    lanes = Lanes()
    lanes.push(_item(1, "A"))
    lanes.pop_ready()
    assert lanes.snapshot() == {"queued": 0, "lanes": 0, "busy_lanes": 1}

    lanes.release("A")

    assert lanes.snapshot() == {"queued": 0, "lanes": 0, "busy_lanes": 0}
    assert "A" not in lanes._lanes


def test_drain_returns_queued_tasks_in_order():
    lanes = Lanes()
    lanes.push(_item(2, "A"))
    lanes.push(_item(1, "B"))
    lanes.push(_item(3, None))
    running = lanes.pop_ready()

    drained = lanes.drain()

    assert running is not None and running.number == 1
    assert [item.number for item in drained] == [2, 3]
    assert lanes.queued == 0
