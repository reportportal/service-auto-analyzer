#  Copyright 2025 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(order=True)
class ProcessingItem:
    """One task to process: a message, or one per-project part of a message.

    Items are ordered by ``priority`` (the publish timestamp) and then by arrival ``number``.
    """

    priority: int
    number: int
    routing_key: str = field(compare=False)
    reply_to: Optional[str] = field(compare=False)
    log_correlation_id: str = field(compare=False)
    msg_correlation_id: str = field(compare=False)
    # Raw JSON body: the coordinator keeps bytes, not parsed objects, so the byte budget matches the memory it holds
    item: bytes = field(compare=False)
    # Project lane of the item, None for items which may run in any order
    lane: Optional[str] = field(default=None, compare=False)
    send_time: Optional[float] = field(default=None, compare=False)
    retries: int = field(default=0, compare=False)

    @property
    def size(self) -> int:
        """Size of the raw body in bytes, as charged to the intake budget."""
        return len(self.item)


@dataclass
class ProcessingResult:
    """Result of one task, sent back by a worker process.

    It refers to the task by its arrival ``number`` instead of carrying the item, so the body isn't sent back.
    """

    number: int
    result: Optional[Any]
    success: bool
    error: Optional[Exception] = None
