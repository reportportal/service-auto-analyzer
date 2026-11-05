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

"""Logging of lost and restored connections of the asynchronous AMQP client."""

import logging
from unittest import mock

import pytest
import pytest_asyncio
from aio_pika.tools import CallbackCollection

from app.amqp.async_amqp import AsyncAmqpClient
from app.commons.model.launch_objects import ApplicationConfig

LOGGER_NAME = "analyzerApp.asyncAmqp"


@pytest.fixture
def connection() -> mock.Mock:
    connection = mock.Mock(is_closed=False, close=mock.AsyncMock())
    connection.close_callbacks = CallbackCollection(connection)
    connection.reconnect_callbacks = CallbackCollection(connection)
    connection.channel = mock.AsyncMock(return_value=mock.Mock(set_qos=mock.AsyncMock()))
    return connection


@pytest_asyncio.fixture
async def client(connection: mock.Mock) -> AsyncAmqpClient:
    client = AsyncAmqpClient(ApplicationConfig(amqpUrl="amqp://user:secret@rabbitmq:5672"))
    with mock.patch("app.amqp.async_amqp.aio_pika.connect_robust", mock.AsyncMock(return_value=connection)):
        await client.connect()
    return client


@pytest.mark.asyncio
async def test_lost_connection_is_logged(client: AsyncAmqpClient, connection: mock.Mock, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        await connection.close_callbacks(ConnectionResetError("reset by peer"))

    assert caplog.messages == ["AMQP connection to amqp://rabbitmq:5672 lost (reset by peer). Reconnecting."]


@pytest.mark.asyncio
async def test_restored_connection_is_logged(client: AsyncAmqpClient, connection: mock.Mock, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        await connection.reconnect_callbacks()

    assert caplog.messages == [
        "AMQP connection to amqp://rabbitmq:5672 restored, the exchange, queue and consumer re-declared."
    ]


@pytest.mark.asyncio
async def test_own_close_is_not_logged_as_lost(client: AsyncAmqpClient, connection: mock.Mock, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER_NAME):
        await client.close()
        await connection.close_callbacks(None)

    connection.close.assert_awaited_once()
    assert caplog.messages == []
