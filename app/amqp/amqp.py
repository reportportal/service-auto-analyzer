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
import uuid
from collections import defaultdict
from collections.abc import Callable
from threading import Lock
from typing import Any, Final, Optional

import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.exceptions import (
    AMQPConnectionError,
    ChannelClosedByBroker,
)
from pika.spec import BasicProperties

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig
from app.utils import url_utils

logger = logging.getLogger("analyzerApp.amqp")

# Maximum back‑off interval when reconnecting (seconds)
ONE_MINUTE: Final[int] = 60
_MAX_SLEEP: Final[int] = ONE_MINUTE

GLOBAL_LOCK: Final[Lock] = Lock()
CREATED_EXCHANGES: Final[dict[str, set[str]]] = defaultdict(set)
EXCHANGE_CREATION_TIME: Final[dict[str, float]] = {}

_DEFAULT_ROUTING_KEY: Final[str] = str(uuid.uuid4())


def do_declare_exchange(
    channel: BlockingChannel, exchange_name: str, exchange_type: str, additional_arguments: Optional[dict[str, Any]]
) -> None:
    channel.exchange_declare(
        exchange=exchange_name,
        exchange_type=exchange_type,
        durable=False,
        auto_delete=True,
        internal=False,
        arguments=additional_arguments,
    )
    logger.info(f"Exchange '{exchange_name}' declared")


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
        self._amqp_base_url_no_credentials = url_utils.remove_credentials_from_url(self._amqp_base_url)
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
        start_time = time.monotonic()
        interval = self._config.amqpInitialRetryInterval
        while True:
            try:
                connection = self._connect()
                logger.info("AMQP connection established.")
                return connection
            except Exception as exc:  # pylint: disable=broad-except
                elapsed = time.monotonic() - start_time
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
        my_connection = self.__connection
        if my_connection is None or my_connection.is_closed:
            my_connection = self._connect_with_retry()
            self.__connection = my_connection
        return my_connection

    def _do_declare_exchange(
        self,
        exchange_name: str,
        exchange_type: str,
        update_interval: int = ONE_MINUTE,
        additional_arguments: Optional[dict[str, Any]] = None,
    ) -> None:
        """Declare application exchange on AMQP server."""
        exchange_key = f"{self._amqp_base_url_no_credentials}:{exchange_name}"
        current_time = time.monotonic()

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
                try:
                    do_declare_exchange(channel, exchange_name, exchange_type, additional_arguments)
                except ChannelClosedByBroker as exc:
                    if (
                        exc.reply_code == 406
                        and "PRECONDITION_FAILED - inequivalent arg 'type' for exchange" in exc.reply_text
                    ):
                        logger.warning(f"Exchange '{exchange_name}' already exists with a different type.")
                        with self._connection.channel() as channel2:
                            channel2.exchange_delete(exchange_name)
                            do_declare_exchange(channel, exchange_name, exchange_type, additional_arguments)

            # Mark that we're creating this exchange
            CREATED_EXCHANGES[self._amqp_base_url_no_credentials].add(exchange_name)
            EXCHANGE_CREATION_TIME[exchange_key] = time.monotonic()

    def _declare_exchange(self) -> None:
        self._do_declare_exchange(
            self._config.amqpExchangeName,
            self._config.amqpExchangeType,
            additional_arguments={
                "analyzer": self._config.amqpExchangeName,
                "analyzer_index": self._config.analyzerIndex,
                "analyzer_priority": self._config.analyzerPriority,
                "analyzer_log_search": self._config.analyzerLogSearch,
                "analyzer_suggest": self._config.analyzerSuggest,
                "analyzer_cluster": self._config.analyzerCluster,
                "version": self._config.appVersion,
            },
        )

    def __send_to_queue(self, exchange_declare_callable: Optional[Callable], **kwargs: Any) -> None:
        while True:
            try:
                # Ensure exchange exists before publishing
                if exchange_declare_callable:
                    exchange_declare_callable()

                with self._connection.channel() as channel:
                    channel.basic_publish(**kwargs)
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

    def publish_response(self, data: str) -> None:
        """Publish an analyzer response message to the configured response queue.

        The response exchange is not declared: ReportPortal owns it and declares it with its own settings, so a
        declaration with other settings is rejected by the broker.

        :param str data: The data to publish
        """
        self.__send_to_queue(
            None,
            exchange=self._config.analyzerResponseExchange,
            routing_key=self._config.analyzerResponseQueue,
            properties=BasicProperties(content_type="application/json"),
            mandatory=False,
            body=bytes(data, "utf-8"),
        )

    def send_to_inner_queue(self, queue: str, data: Any) -> None:
        """Publish message with automatic reconnection.

        :param str queue: Name of the queue to publish to
        :param Any data: Message data to publish
        """
        self.__send_to_queue(
            self._declare_exchange,
            exchange=self._config.amqpExchangeName,
            routing_key=queue,
            body=bytes(data, "utf-8"),
        )

    def reply(self, to: str, correlation_id: str, data: str) -> None:
        """Publish a reply message with automatic reconnection.

        :param str to: The routing key to send the message to
        :param str correlation_id: The correlation ID for the message
        :param str data: The data to publish
        """

        self.__send_to_queue(
            None,
            exchange="",
            routing_key=to,
            properties=BasicProperties(correlation_id=correlation_id, content_type="application/json"),
            mandatory=False,
            body=bytes(data, "utf-8"),
        )
