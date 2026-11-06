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

"""Per-project lanes: tasks of one project run one at a time, in order; tasks of different projects run in parallel."""

import heapq
from typing import Any, Optional

from app.commons.model.processing import ProcessingItem


class Lanes:
    """Queued tasks by project lane.

    A lane is busy from the moment its head is taken for processing until `release` is called for it, retries
    included, so nothing in a project overtakes a running task. Tasks without a lane don't wait for anything.

    Used only from the event loop thread, so there are no locks.
    """

    _lanes: dict[str, list[ProcessingItem]]
    _busy: set[str]
    _free: list[ProcessingItem]

    def __init__(self) -> None:
        self._lanes = {}
        self._busy = set()
        self._free = []

    def push(self, item: ProcessingItem) -> None:
        """Queue a task in its lane, ordered by publish time and arrival number.

        :param item: The task
        """
        if item.lane is None:
            heapq.heappush(self._free, item)
        else:
            heapq.heappush(self._lanes.setdefault(item.lane, []), item)

    def pop_ready(self) -> Optional[ProcessingItem]:
        """Take the oldest task which may run now, and mark its lane busy.

        :return: The task, or None if every queued task waits for a busy lane
        """
        best: Optional[ProcessingItem] = self._free[0] if self._free else None
        for lane, queue in self._lanes.items():
            if queue and lane not in self._busy and (best is None or queue[0] < best):
                best = queue[0]
        if best is None:
            return None
        if best.lane is None:
            return heapq.heappop(self._free)
        self._busy.add(best.lane)
        return heapq.heappop(self._lanes[best.lane])

    def release(self, lane: Optional[str]) -> None:
        """Mark a lane idle after its task finished, and forget the lane if nothing is queued in it.

        :param lane: The lane of the finished task
        """
        if lane is None:
            return
        self._busy.discard(lane)
        if not self._lanes.get(lane):
            self._lanes.pop(lane, None)

    def drain(self) -> list[ProcessingItem]:
        """Remove and return all queued tasks, for shutdown.

        :return: Queued tasks, running ones excluded
        """
        items = list(self._free)
        for queue in self._lanes.values():
            items.extend(queue)
        self._free = []
        self._lanes = {lane: [] for lane in self._busy}
        return sorted(items)

    @property
    def queued(self) -> int:
        """Number of queued tasks, running ones excluded."""
        return len(self._free) + sum(len(queue) for queue in self._lanes.values())

    def snapshot(self) -> dict[str, Any]:
        """State of the lanes, for the health endpoint."""
        return {
            "queued": self.queued,
            "lanes": sum(1 for queue in self._lanes.values() if queue),
            "busy_lanes": len(self._busy),
        }
