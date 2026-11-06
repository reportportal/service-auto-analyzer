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

"""Asynchronous AMQP client of the coordinator: consumes the request queues and publishes replies."""

import asyncio
from typing import Any, Awaitable, Callable, Final, Optional

import aio_pika
from aio_pika.abc import (
    AbstractChannel,
    AbstractConnection,
    AbstractIncomingMessage,
    AbstractQueue,
    AbstractRobustConnection,
    ConsumerTag,
)
from aio_pika.exceptions import ChannelPreconditionFailed
from aio_pika.tools import CallbackType

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.utils import url_utils

LOGGER = logging.getLogger("analyzerApp.asyncAmqp")

MAX_RECONNECT_INTERVAL: Final[int] = 60
PREFETCH_COUNT: Final[int] = 1
JSON_CONTENT_TYPE: Final[str] = "application/json"

MessageCallback = Callable[[AbstractIncomingMessage], Awaitable[Any]]


class ConnectionLostLogger(CallbackType[AbstractConnection, [Optional[BaseException]], None]):
    """Close callback of a robust connection: logs a lost connection. The connection reconnects by itself, every
    5 seconds (the aio-pika default)."""

    _describe: Final[Callable[[], str]]
    _closing: Final[Callable[[], bool]]

    def __init__(self, describe: Callable[[], str], closing: Callable[[], bool]) -> None:
        """Create the callback.

        :param describe: Returns the connection description for the log
        :param closing: Returns whether the client closes the connection itself, which is not logged
        """
        self._describe = describe
        self._closing = closing

    def __call__(self, sender: Optional[AbstractConnection], /, *args: Any, **kwargs: Any) -> None:
        """Log the lost connection.

        :param sender: The connection
        :param args: Why the connection was closed, as the only argument
        """
        if self._closing():
            return
        reason = args[0] if args else None
        LOGGER.warning(f"AMQP connection to {self._describe()} lost ({reason}). Reconnecting.")


class ReconnectedLogger(CallbackType[AbstractRobustConnection, [], None]):
    """Reconnect callback of a robust connection: logs a restored connection, with the exchange, the queue, its
    binding and the consumer declared again."""

    _describe: Final[Callable[[], str]]

    def __init__(self, describe: Callable[[], str]) -> None:
        """Create the callback.

        :param describe: Returns the connection description for the log
        """
        self._describe = describe

    def __call__(self, sender: Optional[AbstractRobustConnection], /, *args: Any, **kwargs: Any) -> None:
        """Log the restored connection.

        :param sender: The connection
        """
        LOGGER.info(f"AMQP connection to {self._describe()} restored, the exchange, queue and consumer re-declared.")


class AsyncAmqpClient:
    """AMQP client on aio-pika: one robust connection, which restores its channel, queue and consumer on reconnect."""

    _config: Final[ApplicationConfig]
    _url: Final[str]
    _url_no_credentials: Final[str]
    _connection: Optional[AbstractRobustConnection]
    _channel: Optional[AbstractChannel]
    _queue: Optional[AbstractQueue]
    _consumer_tag: Optional[ConsumerTag]
    _closing: bool

    def __init__(self, config: ApplicationConfig) -> None:
        """Create the client, it connects on `connect`.

        :param config: The application config object
        """
        self._config = config
        base_url = config.amqpUrl.rstrip("\\/")
        self._url_no_credentials = url_utils.remove_credentials_from_url(base_url)
        self._url = f"{base_url}?heartbeat={config.amqpHeartbeatInterval}"
        self._connection = None
        self._channel = None
        self._queue = None
        self._consumer_tag = None
        self._closing = False

    @property
    def connected(self) -> bool:
        """Whether the connection is open now."""
        return self._connection is not None and not self._connection.is_closed

    async def connect(self) -> None:
        """Connect to the broker, retrying with exponential back-off until it succeeds."""
        interval = self._config.amqpInitialRetryInterval
        while True:
            try:
                LOGGER.info(f"Trying to connect to {self._url_no_credentials}")
                connection = await aio_pika.connect_robust(self._url)
                connection.close_callbacks.add(ConnectionLostLogger(self._describe, lambda: self._closing))
                connection.reconnect_callbacks.add(ReconnectedLogger(self._describe))
                self._connection = connection
                self._channel = await self._open_channel()
                LOGGER.info("AMQP connection established.")
                return
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(f"Connection failed ({exc}). Retrying in {interval} s.")
                LOGGER.debug("Exception details", exc_info=exc)
                await asyncio.sleep(interval)
                interval = min(interval * self._config.amqpBackoffFactor, MAX_RECONNECT_INTERVAL)

    def _describe(self) -> str:
        if self._queue is None:
            return self._url_no_credentials
        return f"{self._url_no_credentials} (queue '{self._queue.name}')"

    async def _open_channel(self) -> AbstractChannel:
        """Open a channel with the prefetch count of the consumer."""
        if self._connection is None:
            raise RuntimeError("AMQP client is not connected")
        channel = await self._connection.channel()
        await channel.set_qos(prefetch_count=PREFETCH_COUNT)
        return channel

    def _get_channel(self) -> AbstractChannel:
        if self._channel is None:
            raise RuntimeError("AMQP client is not connected")
        return self._channel

    async def _declare_exchange(self) -> aio_pika.abc.AbstractExchange:
        """Declare the analyzer exchange, with the analyzer's settings as its arguments for ReportPortal to read."""
        name = self._config.amqpExchangeName
        arguments: dict[str, Any] = {
            "analyzer": name,
            "analyzer_index": self._config.analyzerIndex,
            "analyzer_priority": self._config.analyzerPriority,
            "analyzer_log_search": self._config.analyzerLogSearch,
            "analyzer_suggest": self._config.analyzerSuggest,
            "analyzer_cluster": self._config.analyzerCluster,
            "version": self._config.appVersion,
        }
        kwargs: dict[str, Any] = {"durable": False, "auto_delete": True, "internal": False, "arguments": arguments}
        try:
            exchange = await self._get_channel().declare_exchange(name, self._config.amqpExchangeType, **kwargs)
        except ChannelPreconditionFailed as exc:
            if "inequivalent arg 'type'" not in str(exc):
                raise
            LOGGER.warning(f"Exchange '{name}' already exists with a different type, re-creating it.")
            # A failed declaration closes the channel
            channel = await self._open_channel()
            self._channel = channel
            await channel.exchange_delete(name)
            exchange = await channel.declare_exchange(name, self._config.amqpExchangeType, **kwargs)
        LOGGER.info(f"Exchange '{name}' declared")
        return exchange

    async def consume(self, queue_name: str, callback: MessageCallback) -> None:
        """Declare the exchange and the queue, bind them, and start consuming.

        :param queue_name: Name of the queue to consume
        :param callback: Coroutine function called for every message
        """
        exchange = await self._declare_exchange()
        queue = await self._get_channel().declare_queue(queue_name, durable=True, exclusive=False, auto_delete=False)
        self._queue = queue
        await queue.bind(exchange, routing_key=queue_name)
        self._consumer_tag = await queue.consume(callback, no_ack=False)
        LOGGER.info(f"Start consuming on queue '{queue_name}'")

    async def stop_consuming(self) -> None:
        """Cancel the consumer, so no new messages are delivered."""
        if self._queue is not None and self._consumer_tag is not None:
            try:
                await self._queue.cancel(self._consumer_tag)
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(f"Failed to cancel the consumer cleanly: {exc}")
            self._consumer_tag = None

    async def reply(self, to: str, correlation_id: str, data: str) -> None:
        """Publish a reply to the reply queue of a request.

        :param to: The routing key to send the message to
        :param correlation_id: The correlation ID of the request
        :param data: The data to publish
        """
        message = aio_pika.Message(
            body=data.encode("utf-8"), correlation_id=correlation_id, content_type=JSON_CONTENT_TYPE
        )
        await self._get_channel().default_exchange.publish(message, routing_key=to, mandatory=False)

    async def close(self) -> None:
        """Close the connection."""
        self._closing = True
        if self._connection is not None and not self._connection.is_closed:
            try:
                await self._connection.close()
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(f"Failed to close AMQP connection cleanly: {exc}")
        self._connection = None
        self._channel = None
