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

"""The coordinator which discards the messages of a disabled queue."""

from unittest import mock

import pytest

from app.coordinator.coordinator import DiscardingCoordinator
from app.coordinator.messages import IncomingMessage


def message(routing_key: str) -> IncomingMessage:
    return IncomingMessage(
        routing_key, b'{"project": 1}', None, None, "corr", ack=mock.AsyncMock(), reject=mock.AsyncMock()
    )


@pytest.mark.asyncio
async def test_on_message_acks_and_counts_every_message():
    coordinator = DiscardingCoordinator(name="train", queue="train")
    messages = [message("train_models"), message("index")]

    for incoming in messages:
        await coordinator.on_message(incoming)

    for incoming in messages:
        incoming.ack.assert_awaited_once()
        incoming.reject.assert_not_called()
    assert coordinator.snapshot() == {"name": "train", "connected": True, "discarding": True, "discarded": 2}


@pytest.mark.asyncio
async def test_start_and_shutdown_use_the_client():
    client = mock.Mock(
        connected=False,
        connect=mock.AsyncMock(),
        consume=mock.AsyncMock(),
        stop_consuming=mock.AsyncMock(),
        close=mock.AsyncMock(),
    )
    coordinator = DiscardingCoordinator(name="train", queue="train", client=client)

    await coordinator.start()

    client.connect.assert_awaited_once()
    client.consume.assert_awaited_once()
    assert client.consume.await_args.args[0] == "train"
    assert coordinator.connected is False

    # The consumer callback takes aio-pika messages in
    callback = client.consume.await_args.args[1]
    delivered = mock.Mock(
        routing_key="train_models",
        body=b"{}",
        headers={},
        reply_to=None,
        correlation_id="corr",
        ack=mock.AsyncMock(),
        reject=mock.AsyncMock(),
    )
    await callback(delivered)
    delivered.ack.assert_awaited_once()
    delivered.reject.assert_not_called()
    assert coordinator.snapshot()["discarded"] == 1

    await coordinator.shutdown(grace_period=0)

    client.stop_consuming.assert_awaited_once()
    client.close.assert_awaited_once()
