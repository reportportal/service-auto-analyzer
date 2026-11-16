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

from unittest import mock

import pytest
from opensearchpy import ConflictError, NotFoundError

from app.commons.model.processing import ProcessingItem
from app.coordinator.messages import IncomingMessage, get_priority, retry


def _make_processing_item(routing_key: str) -> ProcessingItem:
    return ProcessingItem(
        priority=10,
        number=1,
        routing_key=routing_key,
        reply_to=None,
        log_correlation_id="log-id",
        msg_correlation_id="msg-id",
        item=b'{"payload": "value"}',
    )


@pytest.mark.parametrize(
    ("exc", "routing_key", "expected"),
    [
        (ConflictError(409, "conflict but no document was found", {}), "item_remove", False),
        (ConflictError(409, "conflict occurred", {}), "item_remove", True),
        (ConflictError(409, "conflict but no document was found", {}), "other_key", True),
        (NotFoundError(404, "resource not found: no such index", {}), "remove_by_launch_start_time", False),
        (NotFoundError(404, "resource not found", {}), "remove_by_launch_start_time", True),
        (NotFoundError(404, "resource not found: no such index", {}), "other_key", True),
        (ValueError("Input X contains NaN, check your data"), "train_models", False),
        (ValueError("unexpected value"), "train_models", True),
        (ValueError("Input X contains NaN, check your data"), "other_key", True),
        (RuntimeError("failure"), "any_key", True),
    ],
)
def test_retry_returns_expected_value(exc: Exception, routing_key: str, expected: bool) -> None:
    item = _make_processing_item(routing_key)
    result = retry(item, exc)

    assert result is expected


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"timestamp_in_ms": 1749029201296}, 1749029201296),
        ({"timestamp_in_ms": "1749029201296L"}, 1749029201296),
        ({"timestamp_in_ms": "1749029201296"}, 1749029201296),
    ],
)
def test_priority_from_header(headers, expected):
    assert get_priority(headers) == expected


@pytest.mark.parametrize("headers", [None, {}, {"other": 1}, {"timestamp_in_ms": "not a number"}])
def test_priority_defaults_to_current_time(headers, monkeypatch):
    monkeypatch.setattr("app.coordinator.messages.time.time", lambda: 1234.5)

    assert get_priority(headers) == 1234500


@pytest.mark.asyncio
async def test_incoming_message_from_aio_pika():
    delivered = mock.Mock(
        routing_key="suggest",
        body=b'{"project": 1}',
        headers={"timestamp_in_ms": 1},
        reply_to="reply-queue",
        correlation_id="corr",
        ack=mock.AsyncMock(),
        reject=mock.AsyncMock(),
    )

    message = IncomingMessage.from_aio_pika(delivered)

    assert (message.routing_key, message.body, message.headers, message.reply_to, message.correlation_id) == (
        "suggest",
        b'{"project": 1}',
        {"timestamp_in_ms": 1},
        "reply-queue",
        "corr",
    )
    await message.ack()
    delivered.ack.assert_awaited_once_with()
    await message.reject()
    delivered.reject.assert_awaited_once_with(requeue=False)


def test_incoming_message_from_aio_pika_without_routing_key():
    delivered = mock.Mock(routing_key=None, ack=mock.AsyncMock(), reject=mock.AsyncMock())

    assert IncomingMessage.from_aio_pika(delivered).routing_key == ""
