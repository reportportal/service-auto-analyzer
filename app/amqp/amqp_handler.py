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

import json

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.service.processor import Processor

logger = logging.getLogger("analyzerApp.amqpHandler")


class AmqpRequestHandler:
    """Class for handling AMQP requests with service routing based on routing key"""

    def __init__(self, app_config: ApplicationConfig, search_config: SearchConfig):
        """Initialize processor for handling requests"""
        self.app_config = app_config
        self.search_config = search_config
        self._processor = Processor(app_config, search_config)

    def handle_amqp_request(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        props: BasicProperties,
        body: bytes,
    ) -> None:
        """Function for handling amqp request: index, search and analyze using routing key."""
        routing_key = method.routing_key

        # Processing request
        logging.new_correlation_id()
        logger.debug(f"Processing message: --Method: {method} --Properties: {props} --Body: {body}")
        try:
            message = json.loads(body, strict=False)
            channel.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as exc:
            logger.exception("Failed to parse message body to JSON", exc_info=exc)
            channel.basic_nack(delivery_tag=method.delivery_tag, requeue=False)
            return None

        # Process using the processor
        try:
            response_body = self._processor.process(routing_key, message)
        except Exception as exc:
            logger.exception("Failed to process message", exc_info=exc)
            return None

        # Sending response if applicable
        if response_body is None:
            return None

        try:
            if props.reply_to:
                channel.basic_publish(
                    exchange="",
                    routing_key=props.reply_to,
                    properties=BasicProperties(correlation_id=props.correlation_id, content_type="application/json"),
                    mandatory=False,
                    body=bytes(response_body, "utf-8"),
                )
        except Exception as exc:
            logger.exception("Failed to publish result", exc_info=exc)
            return None
        logger.debug("Finished processing response")
        return None
