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

"""Publishing of the synchronous AMQP client which workers use."""

from unittest import mock

import pytest

from app.amqp.amqp import AmqpClient
from app.commons.model.launch_objects import ApplicationConfig


@pytest.fixture
def channel() -> mock.MagicMock:
    channel = mock.MagicMock()
    channel.__enter__.return_value = channel
    return channel


@pytest.fixture
def client(channel: mock.MagicMock) -> AmqpClient:
    connection = mock.Mock(is_closed=False)
    connection.channel.return_value = channel
    client = AmqpClient(ApplicationConfig(amqpUrl="amqp://user:secret@rabbitmq:5672/analyzer"))
    with mock.patch("app.amqp.amqp.pika.BlockingConnection", return_value=connection):
        client._connection  # Connect once, the connection is reused
    return client


def test_publish_response_does_not_declare_the_exchange(client: AmqpClient, channel: mock.MagicMock):
    client.publish_response("[]")

    channel.exchange_declare.assert_not_called()
    channel.basic_publish.assert_called_once()
    kwargs = channel.basic_publish.call_args.kwargs
    assert kwargs["exchange"] == "analyzer-reply"
    assert kwargs["routing_key"] == "analysis.matches"
    assert kwargs["body"] == b"[]"
