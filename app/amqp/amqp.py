#  Copyright 2023 EPAM Systems
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

import os
import time
from collections.abc import Callable

import pika
from pika.adapters.blocking_connection import BlockingChannel, BlockingConnection
from pika.spec import Basic, BasicProperties

from app.commons import logging
from app.utils import text_processing

logger = logging.getLogger("analyzerApp.amqp")


class AmqpClient:
    """AmqpClient handles communication with rabbitmq"""

    connection: BlockingConnection

    def __init__(self, amqp_url: str, retry_interval: int = 10, max_retry_time: int = 300) -> None:
        """Initialize the AMQP client with retry mechanism
        
        Args:
            amqp_url: The AMQP URL to connect to
            retry_interval: Time in seconds between connection retry attempts (default: 10)
            max_retry_time: Maximum time in seconds to keep retrying (default: 300)
        """
        self.connection = self.create_ampq_connection_with_retry(
            amqp_url, retry_interval, max_retry_time)

    @staticmethod
    def create_ampq_connection(amqp_url: str) -> BlockingConnection:
        """Creates AMQP client"""
        amqp_full_url = amqp_url.rstrip("\\").rstrip("/") + "?heartbeat=600"
        logger.info(f"Try connect to {text_processing.remove_credentials_from_url(amqp_full_url)}")
        return pika.BlockingConnection(pika.connection.URLParameters(amqp_full_url))
    
    @staticmethod
    def create_ampq_connection_with_retry(amqp_url: str, retry_interval: int = 10, 
                                          max_retry_time: int = 300) -> BlockingConnection:
        """Creates AMQP client with retry mechanism
        
        Args:
            amqp_url: The AMQP URL to connect to
            retry_interval: Time in seconds between connection retry attempts
            max_retry_time: Maximum time in seconds to keep retrying
            
        Returns:
            BlockingConnection: The AMQP connection
            
        Raises:
            RuntimeError: If connection could not be established after max retry time
        """
        start_time = time.time()
        last_exception = None
        
        while time.time() - start_time < max_retry_time:
            try:
                connection = AmqpClient.create_ampq_connection(amqp_url)
                logger.info("Successfully established AMQP connection")
                return connection
            except Exception as exc:
                last_exception = exc
                logger.error("Failed to connect to AMQP, retrying in %d seconds...", retry_interval)
                logger.debug("Connection error details: %s", str(exc))
                time.sleep(retry_interval)
        
        # If we get here, we've exceeded the maximum retry time
        logger.error("Failed to establish AMQP connection after %d seconds", max_retry_time)
        if last_exception:
            logger.exception("Last connection error", exc_info=last_exception)
        raise RuntimeError(f"Could not establish AMQP connection after {max_retry_time} seconds")

    @staticmethod
    def bind_queue(channel: BlockingChannel, name: str, exchange_name: str) -> bool:
        """AmqpClient binds a queue with an exchange for rabbitmq"""
        try:
            result = channel.queue_declare(queue=name, durable=False, exclusive=False, auto_delete=True,
                                           arguments=None)
        except Exception as exc:
            logger.exception(f'Failed to declare a queue "{name}" pid({os.getpid()})', exc_info=exc)
            os.kill(os.getpid(), 9)
            return False
        logger.info("Queue '%s' has been declared pid(%d)", result.method.queue, os.getpid())
        try:
            channel.queue_bind(exchange=exchange_name, queue=result.method.queue, routing_key=name)
        except Exception as exc:
            logger.exception(f'Failed to bind a queue "{name}" pid({os.getpid()})', exc_info=exc)
            os.kill(os.getpid(), 9)
        return True

    @staticmethod
    def consume_queue(channel: BlockingChannel, queue: str, auto_ack: bool, exclusive: bool,
                      msg_callback: Callable[[
                          BlockingChannel,
                          Basic.Deliver,
                          BasicProperties,
                          bytes,
                      ], None]) -> None:
        """AmqpClient shows how to handle a message from the queue"""
        try:
            channel.basic_qos(prefetch_count=1, prefetch_size=0)
        except Exception as exc:
            logger.exception(f"Failed to configure Qos pid({os.getpid()})", exc_info=exc)
            os.kill(os.getpid(), 9)
        try:
            channel.basic_consume(queue=queue, auto_ack=auto_ack, exclusive=exclusive, on_message_callback=msg_callback)
        except Exception as exc:
            logger.exception(f"Failed to register a consumer pid({os.getpid()})", exc_info=exc)
            os.kill(os.getpid(), 9)

    def receive(self, exchange_name: str, queue: str, auto_ack: bool, exclusive: bool,
                msg_callback: Callable[[
                    BlockingChannel,
                    Basic.Deliver,
                    BasicProperties,
                    bytes,
                ], None]) -> None:
        """AmqpClient starts consuming messages from a specific queue"""
        try:
            channel = self.connection.channel()
            AmqpClient.bind_queue(channel, queue, exchange_name)
            AmqpClient.consume_queue(channel, queue, auto_ack, exclusive, msg_callback)
            logger.info("started consuming pid(%d) on the queue %s", os.getpid(), queue)
            channel.start_consuming()
        except Exception as exc:
            logger.exception(f"Failed to consume messages pid({os.getpid()}) in queue '{queue}'", exc_info=exc)
            os.kill(os.getpid(), 9)

    def send_to_inner_queue(self, exchange_name: str, queue: str, data: str) -> None:
        try:
            channel = self.connection.channel()
            channel.basic_publish(exchange=exchange_name, routing_key=queue, body=bytes(data, 'utf-8'))
        except Exception as exc:
            logger.exception(f"Failed to publish messages in queue '{queue}'", exc_info=exc)

    def close(self) -> None:
        """AmqpClient closes the connection"""
        try:
            self.connection.close()
        except Exception as exc:
            logger.error("Failed to close connection")
            logger.exception("Failed to close connection", exc_info=exc)
