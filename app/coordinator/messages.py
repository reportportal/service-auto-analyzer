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

"""Message helpers of the coordinator: the incoming message, priority, retry policy and message logging."""

import json
import time
from dataclasses import dataclass
from logging import DEBUG
from typing import Any, Awaitable, Callable, Mapping, Optional

from aio_pika.abc import AbstractIncomingMessage
from opensearchpy import ConflictError, NotFoundError

from app.commons import logging
from app.commons.model.processing import ProcessingItem

LOGGER = logging.getLogger("analyzerApp.coordinator")

TIMESTAMP_HEADER = "timestamp_in_ms"

Acknowledge = Callable[[], Awaitable[None]]


@dataclass(frozen=True)
class IncomingMessage:
    """What the coordinators need from an AMQP message."""

    routing_key: str
    body: bytes
    headers: Optional[Mapping[str, Any]]
    reply_to: Optional[str]
    correlation_id: Optional[str]
    ack: Acknowledge
    """Acknowledges the message"""
    reject: Acknowledge
    """Rejects the message without requeueing it"""

    @classmethod
    def from_aio_pika(cls, message: AbstractIncomingMessage) -> "IncomingMessage":
        """Take what the coordinators need from an aio-pika message.

        :param message: The aio-pika message
        :return: The incoming message
        """

        async def reject() -> None:
            await message.reject(requeue=False)

        return cls(
            routing_key=message.routing_key or "",
            body=message.body,
            headers=message.headers,
            reply_to=message.reply_to,
            correlation_id=message.correlation_id,
            ack=message.ack,
            reject=reject,
        )


def get_priority(headers: Optional[Mapping[str, Any]]) -> int:
    """Get the priority of a message: its publish time from the `timestamp_in_ms` header, or the current time.

    :param headers: Message headers
    :return: Priority, the lower the earlier
    """
    # Wall clock, not monotonic: the fallback is compared with the publish timestamps of other messages, which are
    # epoch milliseconds
    priority = int(time.time() * 1000)
    if headers and TIMESTAMP_HEADER in headers:
        try:
            timestamp_str = str(headers[TIMESTAMP_HEADER])
            # Remove 'L' suffix if present (e.g., 1749029201296L)
            if timestamp_str.endswith("L"):
                timestamp_str = timestamp_str[:-1]
            priority = int(timestamp_str)
        except (ValueError, TypeError) as exc:
            LOGGER.warning(f"Failed to parse timestamp_in_ms header: {headers[TIMESTAMP_HEADER]}", exc_info=exc)
    return priority


def retry(item: ProcessingItem, exc: Optional[Exception]) -> bool:
    """Default retry predicate: whether a failed task should be retried.

    :param item: The failed task
    :param exc: The error it failed with
    :return: True to retry
    """
    if exc is None:
        return True
    if isinstance(exc, ConflictError) and item.routing_key == "item_remove":
        return "but no document was found" not in exc.error
    if isinstance(exc, NotFoundError) and item.routing_key == "remove_by_launch_start_time":
        return "no such index" not in exc.error
    if isinstance(exc, ValueError) and item.routing_key == "train_models":
        return "Input X contains NaN" not in str(exc)
    return True


def log_incoming_message(item: ProcessingItem) -> None:
    if LOGGER.is_enabled_for(DEBUG):
        LOGGER.debug(
            f"Processing message: --Routing key: {item.routing_key} --Correlation ID: {item.msg_correlation_id} "
            f"--Lane: {item.lane} --Body: {item.item.decode('utf-8', errors='replace')}"
        )


def log_outgoing_message(reply_to: str, correlation_id: str, body: Any) -> None:
    if LOGGER.is_enabled_for(DEBUG):
        LOGGER.debug(
            f"Replying message: --To: {reply_to} --Correlation ID: {correlation_id} --Body: {json.dumps(body)}"
        )


def log_published_message(exchange: str, routing_key: str, body: str) -> None:
    if LOGGER.is_enabled_for(DEBUG):
        LOGGER.debug(f"Publishing message: --Exchange: {exchange} --Routing key: {routing_key} --Body: {body}")
