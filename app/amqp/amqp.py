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

import pika

from app.commons import logging
from app.utils import text_processing

logger = logging.getLogger("analyzerApp.amqp")


class AmqpClient:
    """AmqpClient handles communication with rabbitmq"""

    def __init__(self, amqp_url):
        self.connection = AmqpClient.create_ampq_connection(amqp_url)

    @staticmethod
    def create_ampq_connection(amqp_url):
        """Creates AMQP client"""
        amqp_full_url = amqp_url.rstrip("\\").rstrip("/") + "?heartbeat=600"
        logger.info("Try connect to %s" % text_processing.remove_credentials_from_url(amqp_full_url))
        return pika.BlockingConnection(pika.connection.URLParameters(amqp_full_url))

    @staticmethod
    def bind_queue(channel, name, exchange_name):
        """AmqpClient binds a queue with an exchange for rabbitmq"""
        try:
            result = channel.queue_declare(queue=name, durable=False, exclusive=False, auto_delete=True,
                                           arguments=None)
        except Exception as exc:
            logger.error(f'Failed to declare a queue "{name}" pid({os.getpid()})')
            logger.exception(exc)
            os.kill(os.getpid(), 9)
            return False
        logger.info("Queue '%s' has been declared pid(%d)", result.method.queue, os.getpid())
        try:
            channel.queue_bind(exchange=exchange_name, queue=result.method.queue, routing_key=name)
        except Exception as exc:
            logger.error(f'Failed to bind a queue "{name}" pid({os.getpid()})')
            logger.exception(exc)
            os.kill(os.getpid(), 9)
        return True

    @staticmethod
    def consume_queue(channel, queue, auto_ack, exclusive, msg_callback):
        """AmqpClient shows how to handle a message from the queue"""
        try:
            channel.basic_qos(prefetch_count=1, prefetch_size=0)
        except Exception as exc:
            logger.error("Failed to configure Qos pid(%d)", os.getpid())
            logger.exception(exc)
            os.kill(os.getpid(), 9)
        try:
            channel.basic_consume(queue=queue, auto_ack=auto_ack, exclusive=exclusive,
                                  on_message_callback=msg_callback)
        except Exception as exc:
            logger.error("Failed to register a consumer pid(%d)", os.getpid())
            logger.exception(exc)
            os.kill(os.getpid(), 9)

    def receive(self, exchange_name, queue, auto_ack, exclusive, msg_callback):
        """AmqpClient starts consuming messages from a specific queue"""
        try:
            channel = self.connection.channel()
            AmqpClient.bind_queue(channel, queue, exchange_name)
            AmqpClient.consume_queue(channel, queue, auto_ack, exclusive, msg_callback)
            logger.info("started consuming pid(%d) on the queue %s", os.getpid(), queue)
            channel.start_consuming()
        except Exception as exc:
            logger.error("Failed to consume messages pid(%d) in queue %s", os.getpid(), queue)
            logger.exception(exc)
            os.kill(os.getpid(), 9)

    def send_to_inner_queue(self, exchange_name, queue, data):
        try:
            channel = self.connection.channel()
            channel.basic_publish(
                exchange=exchange_name,
                routing_key=queue,
                body=data)
        except Exception as exc:
            logger.error("Failed to publish messages in queue %s", queue)
            logger.exception(exc)
