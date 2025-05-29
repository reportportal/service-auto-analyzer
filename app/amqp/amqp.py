#  Copyright 2025 EPAM Systems
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

import time
from collections import defaultdict
from collections.abc import Callable
from threading import Lock
from typing import Final, Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.exceptions import (
    AMQPConnectionError,
    ChannelClosedByBroker,
)
from pika.spec import Basic, BasicProperties

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.utils import text_processing

logger = logging.getLogger("analyzerApp.amqp")

# Maximum back‑off interval when reconnecting (seconds)
ONE_MINUTE: Final[int] = 60
_MAX_SLEEP: Final[int] = ONE_MINUTE

GLOBAL_LOCK: Final[Lock] = Lock()
CREATED_EXCHANGES: Final[dict[str, set[str]]] = defaultdict(set)
EXCHANGE_CREATION_TIME: Final[dict[str, float]] = dict()


class AmqpClientConnectionException(Exception):
    """Exception raised when AMQP connection fails."""

    def __init__(self, message: str) -> None:
        """Initialize the exception with a message.

        :param str message: Error message
        """
        super().__init__(message)
        self.message = message


class AmqpClient:
    """AMQP client wrapper able to recover from transient network failures."""

    _config: Final[ApplicationConfig]
    _amqp_base_url: Final[str]
    _amqp_base_url_no_credentials: Final[str]
    _amqp_url: Final[str]
    __connection: Optional[BlockingConnection] = None

    def __init__(self, config: ApplicationConfig) -> None:
        """Initialize the AMQP client with retry mechanism.

        :param ApplicationConfig config: the application config object
        """
        self._config = config
        self._amqp_base_url = config.amqpUrl.rstrip("\\/")
        self._amqp_base_url_no_credentials = text_processing.remove_credentials_from_url(self._amqp_base_url)
        self._amqp_url = self._amqp_base_url + f"?heartbeat={config.amqpHeartbeatInterval}"

    def close(self) -> None:
        """Close the connection if it is opened."""
        try:
            if self.__connection and self.__connection.is_open:
                self.__connection.close()
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning(f"Failed to close AMQP connection cleanly: {exc}")
        self.__connection = None

    def _connect(self) -> BlockingConnection:
        """Creates AMQP connection.

        :return: The AMQP connection
        """
        logger.info(f"Trying to connect to {self._amqp_base_url_no_credentials}")
        parameters = pika.connection.URLParameters(self._amqp_url)
        return pika.BlockingConnection(parameters)

    def _connect_with_retry(self) -> BlockingConnection:
        """Attempt to establish connection using exponential back‑off.

        :raises AmqpClientConnectionException: If connection could not be established after max retry time
        :return: The AMQP connection
        """
        start_time = time.time()
        interval = self._config.amqpInitialRetryInterval
        while True:
            try:
                connection = self._connect()
                logger.info("AMQP connection established.")
                return connection
            except Exception as exc:  # pylint: disable=broad-except
                elapsed = time.time() - start_time
                if elapsed >= self._config.amqpMaxRetryTime:
                    logger.error(f"Exceeded max retry time ({self._config.amqpMaxRetryTime} s).")
                    raise AmqpClientConnectionException("Could not establish AMQP connection") from exc
                logger.warning(f"Connection failed ({exc}). Retrying in {interval} s.")
                logger.debug("Exception details", exc_info=exc)
                time.sleep(interval)
                interval = min(interval * self._config.amqpBackoffFactor, _MAX_SLEEP)

    @property
    def _connection(self) -> BlockingConnection:
        """Get the current AMQP connection.

        :return: The AMQP connection
        """
        if self.__connection is None or self.__connection.is_closed:
            self.__connection = self._connect_with_retry()
        return self.__connection

    def _declare_exchange(self, update_interval: int = ONE_MINUTE) -> None:
        """Declare application exchange on AMQP server."""
        exchange_name = self._config.amqpExchangeName
        exchange_key = f"{self._amqp_base_url_no_credentials}:{exchange_name}"
        current_time = time.time()

        # Check if exchange was created recently (within 1 minute)
        if (
            exchange_key in EXCHANGE_CREATION_TIME
            and current_time - EXCHANGE_CREATION_TIME[exchange_key] < update_interval
        ):
            logger.debug(f"Exchange '{exchange_name}' was recently created, skipping declaration")
            return

        with GLOBAL_LOCK:
            # Check if exchange was created recently (within 1 minute) once again after acquiring the lock
            if (
                exchange_key in EXCHANGE_CREATION_TIME
                and current_time - EXCHANGE_CREATION_TIME[exchange_key] < update_interval
            ):
                logger.debug(f"Exchange '{exchange_name}' was recently created, skipping declaration")
                return

            with self._connection.channel() as channel:
                channel.exchange_declare(
                    exchange=exchange_name,
                    exchange_type="direct",
                    durable=False,
                    auto_delete=True,
                    internal=False,
                    arguments={
                        "analyzer": exchange_name,
                        "analyzer_index": self._config.analyzerIndex,
                        "analyzer_priority": self._config.analyzerPriority,
                        "analyzer_log_search": self._config.analyzerLogSearch,
                        "analyzer_suggest": self._config.analyzerSuggest,
                        "analyzer_cluster": self._config.analyzerCluster,
                        "version": self._config.appVersion,
                    },
                )
                logger.info(f"Exchange '{exchange_name}' declared")

            # Mark that we're creating this exchange
            CREATED_EXCHANGES[self._amqp_base_url_no_credentials].add(exchange_name)
            EXCHANGE_CREATION_TIME[exchange_key] = time.time()

    @staticmethod
    def _bind_queue(channel: BlockingChannel, name: str, exchange_name: str) -> None:
        """Bind to a queue and exchange.

        :param BlockingChannel channel: The channel to bind the queue on
        :param str name: Name of the queue
        :param str exchange_name: Name of the exchange
        """
        channel.queue_declare(queue=name, durable=True, exclusive=False, auto_delete=False, arguments=None)
        channel.queue_bind(exchange=exchange_name, queue=name, routing_key=name)

    @staticmethod
    def _consume_queue(
        channel: BlockingChannel,
        queue: str,
        auto_ack: bool,
        exclusive: bool,
        msg_callback: Callable[[BlockingChannel, Basic.Deliver, BasicProperties, bytes], None],
    ) -> None:
        """Signal to consume messages from the queue and bind to message callback.

        :param BlockingChannel channel: The channel to consume from
        :param str queue: Name of the queue to consume
        :param bool auto_ack: Whether to automatically acknowledge messages
        :param bool exclusive: Whether to set exclusive consumer
        :param callable msg_callback: Callback function to handle received messages
        """
        channel.basic_qos(prefetch_count=1, prefetch_size=0)
        channel.basic_consume(
            queue=queue,
            auto_ack=auto_ack,
            exclusive=exclusive,
            on_message_callback=msg_callback,
        )

    def receive(
        self,
        queue: str,
        msg_callback: Callable[[BlockingChannel, Basic.Deliver, BasicProperties, bytes], None],
    ) -> None:
        """Continuously consume messages, reconnect on failure.

        :param str queue: Name of the queue to consume
        :param callable msg_callback: Callback function to handle received messages
        """
        connection_info = f"Exchange: '{self._config.amqpExchangeName}'. Queue: '{queue}'."
        while True:
            try:
                # Ensure exchange exists before consuming
                self._declare_exchange()

                with self._connection.channel() as channel:
                    self._bind_queue(channel, queue, self._config.amqpExchangeName)
                    self._consume_queue(channel, queue, False, False, msg_callback)
                    logger.info(f"Start consuming on queue '{queue}'")
                    channel.start_consuming()
            except (AMQPConnectionError, ChannelClosedByBroker) as exc:
                logger.exception(f"Connection/channel lost. Reconnecting. {connection_info}", exc_info=exc)
                self.close()
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception(f"Unexpected error in consumer. Reconnecting. {connection_info}", exc_info=exc)
                self.close()
            except KeyboardInterrupt:
                logger.info(f"Consumer interrupted by user. Exiting. {connection_info}")
                break

    def send_to_inner_queue(self, queue: str, data: str) -> None:
        """Publish message with automatic reconnection.

        :param str queue: Name of the queue to publish to
        :param str data: Message data to publish
        """
        while True:
            try:
                # Ensure exchange exists before publishing
                self._declare_exchange()

                with self._connection.channel() as channel:
                    channel.basic_publish(
                        exchange=self._config.amqpExchangeName,
                        routing_key=queue,
                        body=data.encode("utf‑8"),
                    )
                return  # success
            except AMQPConnectionError as exc:
                logger.warning(f"Publish failed: {exc}. Reconnecting.", exc_info=exc)
                self.close()
            except AmqpClientConnectionException:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                logger.exception("Failed to publish message", exc_info=exc)
                self.close()
            except KeyboardInterrupt:
                logger.info("Consumer interrupted by user. Exiting.")
                break
