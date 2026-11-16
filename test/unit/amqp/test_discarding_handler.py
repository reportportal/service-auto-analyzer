#  Copyright 2026 EPAM Systems
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

"""The handler which discards the messages of the "train" queue when training is disabled."""

import logging
from typing import Optional
from unittest import mock

import pytest

from app.amqp.amqp_handler import DiscardingAmqpRequestHandler


def deliver(routing_key: str, delivery_tag: Optional[int]) -> mock.Mock:
    return mock.Mock(routing_key=routing_key, delivery_tag=delivery_tag)


def test_every_message_is_acked_and_counted():
    handler = DiscardingAmqpRequestHandler(name="train_handler")
    channel = mock.Mock()

    handler.handle_amqp_request(channel, deliver("train_models", 1), mock.Mock(), b'{"project": 1}')
    handler.handle_amqp_request(channel, deliver("index", 2), mock.Mock(), b"[]")

    assert channel.basic_ack.call_args_list == [mock.call(delivery_tag=1), mock.call(delivery_tag=2)]
    channel.basic_nack.assert_not_called()
    channel.basic_publish.assert_not_called()
    assert handler.discarded == 2


def test_malformed_message_is_acked_without_parsing():
    handler = DiscardingAmqpRequestHandler(name="train_handler")
    channel = mock.Mock()

    handler.handle_amqp_request(channel, deliver("train_models", 7), mock.Mock(), b"{not json")

    channel.basic_ack.assert_called_once_with(delivery_tag=7)
    channel.basic_nack.assert_not_called()
    assert handler.discarded == 1


def test_message_without_delivery_tag_is_counted():
    handler = DiscardingAmqpRequestHandler(name="train_handler")
    channel = mock.Mock()

    handler.handle_amqp_request(channel, deliver("train_models", None), mock.Mock(), b"{}")

    channel.basic_ack.assert_not_called()
    assert handler.discarded == 1


def test_health_attributes():
    handler = DiscardingAmqpRequestHandler(name="train_handler")

    assert handler.running_tasks == []
    assert handler.processor.is_alive() is True
    assert isinstance(handler.processor.pid, int)


def test_shutdown_logs_discarded_count(caplog: pytest.LogCaptureFixture):
    handler = DiscardingAmqpRequestHandler(name="train_handler")
    handler.handle_amqp_request(mock.Mock(), deliver("train_models", 1), mock.Mock(), b"{}")

    with caplog.at_level(logging.INFO, logger="analyzerApp.amqpHandler"):
        handler.shutdown()

    assert caplog.messages == ["Handler 'train_handler' stopped, 1 messages discarded"]
