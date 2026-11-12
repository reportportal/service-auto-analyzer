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

"""The replies of a finished task: the coordinator sends each one where its type says."""

import asyncio
import json
import logging
from typing import Optional
from unittest.mock import AsyncMock

import pytest

from app.commons.model.launch_objects import ApplicationConfig
from app.commons.model.processing import Reply, ReplyType
from app.coordinator.budget import IntakeBudget
from app.coordinator.coordinator import LaneCoordinator
from app.coordinator.messages import IncomingMessage
from test.unit.coordinator.test_coordinator_retries import FakePool

APP_CONFIG = ApplicationConfig(
    amqpExchangeName="analyzer", analyzerResponseExchange="analyzer-reply", analyzerResponseQueue="analysis.matches"
)
REPLIES = [
    Reply("[1]"),
    Reply('[{"testItem": 1}]', ReplyType.ASYNC_REPLY),
    Reply('{"project": 1}', ReplyType.QUEUE_SEND, "train_models"),
    Reply('{"project": 2}', ReplyType.QUEUE_SEND, "train_models"),
]


def make_coordinator(pool: FakePool, reply: AsyncMock, publish: AsyncMock) -> LaneCoordinator:
    return LaneCoordinator(
        APP_CONFIG,
        name="test",
        queue="all",
        routing_key_predicate=lambda _: True,
        pool=pool,
        budget=IntakeBudget(10, 1000),
        reply=reply,
        publish=publish,
    )


def message(reply_to: Optional[str]) -> IncomingMessage:
    body = json.dumps({"project": 1}).encode("utf-8")
    return IncomingMessage("suggest", body, None, reply_to, "corr", ack=AsyncMock(), reject=AsyncMock())


async def finish(pool: FakePool, replies: Optional[list[Reply]]) -> None:
    pool.finish(success=True, replies=replies)
    # Let the background send run
    for _ in range(5):
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_each_reply_is_sent_where_its_type_says():
    pool, reply, publish = FakePool(), AsyncMock(), AsyncMock()
    await make_coordinator(pool, reply, publish).on_message(message("reply-queue"))

    await finish(pool, REPLIES)

    reply.assert_awaited_once_with("reply-queue", "corr", "[1]")
    assert publish.await_args_list == [
        (("analyzer-reply", "analysis.matches", '[{"testItem": 1}]'),),
        (("analyzer", "train_models", '{"project": 1}'),),
        (("analyzer", "train_models", '{"project": 2}'),),
    ]


@pytest.mark.asyncio
async def test_direct_reply_needs_reply_to():
    pool, reply, publish = FakePool(), AsyncMock(), AsyncMock()
    await make_coordinator(pool, reply, publish).on_message(message(None))

    await finish(pool, REPLIES)

    reply.assert_not_awaited()
    assert publish.await_count == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("replies", [None, []])
async def test_nothing_to_send(replies: Optional[list[Reply]]):
    pool, reply, publish = FakePool(), AsyncMock(), AsyncMock()
    await make_coordinator(pool, reply, publish).on_message(message("reply-queue"))

    await finish(pool, replies)

    reply.assert_not_awaited()
    publish.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_send_does_not_stop_the_others(caplog: pytest.LogCaptureFixture):
    pool, reply, publish = FakePool(), AsyncMock(side_effect=RuntimeError("reply failed")), AsyncMock()
    publish.side_effect = [RuntimeError("response failed"), None, None]
    await make_coordinator(pool, reply, publish).on_message(message("reply-queue"))

    with caplog.at_level(logging.ERROR, logger="analyzerApp.coordinator"):
        await finish(pool, REPLIES)

    reply.assert_awaited_once()
    assert publish.await_count == 3
    assert caplog.messages == ["Failed to send the DIRECT_REPLY reply", "Failed to send the ASYNC_REPLY reply"]
