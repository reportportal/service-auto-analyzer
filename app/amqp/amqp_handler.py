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
import queue
import threading
import time
from typing import Any, Callable, Optional

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.amqp.amqp import AmqpClient
from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem
from app.commons.processing import Processor
from app.service.processor import ServiceProcessor

logger = logging.getLogger("analyzerApp.amqpHandler")


def log_incoming_message(routing_key: str, correlation_id: str, body: Any) -> None:
    logging.new_correlation_id()
    body = json.dumps(body)
    logger.debug(f"Processing message: --Routing key: {routing_key} --Correlation ID: {correlation_id} --Body: {body}")


def log_outgoing_message(reply_to: str, correlation_id: str, body: Any) -> None:
    logger.debug(f"Replying message: --To: {reply_to} --Correlation ID: {correlation_id} --Body: {json.dumps(body)}")


def serialize_message(channel: BlockingChannel, delivery_tag: Optional[int], body: bytes) -> Optional[Any]:
    try:
        message = json.loads(body, strict=False)
        if delivery_tag is not None:
            channel.basic_ack(delivery_tag=delivery_tag)
        return message
    except Exception as exc:
        logger.exception("Failed to parse message body to JSON", exc_info=exc)
        if delivery_tag is not None:
            channel.basic_nack(delivery_tag=delivery_tag, requeue=False)
        return None


def get_priority(props: BasicProperties) -> int:
    """Determine priority from timestamp_in_ms header or use current time."""
    priority = int(time.time() * 1000)  # Default to current time in milliseconds
    if props.headers and "timestamp_in_ms" in props.headers:
        try:
            timestamp_str = str(props.headers["timestamp_in_ms"])
            # Remove 'L' suffix if present (e.g., 1749029201296L)
            if timestamp_str.endswith("L"):
                timestamp_str = timestamp_str[:-1]
            priority = int(timestamp_str)
        except (ValueError, TypeError) as exc:
            logger.warning(f"Failed to parse timestamp_in_ms header: {props.headers['timestamp_in_ms']}", exc_info=exc)
    return priority


class AtomicInteger:
    def __init__(self, value: int = 0) -> None:
        self._value = int(value)
        self._lock = threading.Lock()

    def inc(self, d: int = 1) -> int:
        with self._lock:
            self._value += int(d)
            return self._value

    def dec(self, d: int = 1) -> int:
        return self.inc(-d)

    @property
    def value(self) -> int:
        with self._lock:
            return self._value


class ProcessAmqpRequestHandler:
    """Class for handling AMQP requests with process-based routing and priority queue"""

    app_config: ApplicationConfig
    search_config: SearchConfig
    client: AmqpClient
    queue_size: int
    prefetch_size: int
    routing_key_predicate: Optional[Callable[[str], bool]]
    counter: AtomicInteger
    queue: queue.PriorityQueue[ProcessingItem]
    running_tasks: list[ProcessingItem]
    processor: Processor
    _processing_thread: Optional[threading.Thread]
    _shutdown: bool

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        queue_size: int = 100,
        prefetch_size: int = 2,
        routing_key_predicate: Optional[Callable[[str], bool]] = None,
    ):
        """Initialize processor for handling requests with process-based communication"""
        self.app_config = app_config
        self.search_config = search_config
        self.client = AmqpClient(app_config)
        self.queue_size = queue_size
        self.prefetch_size = prefetch_size
        self.routing_key_predicate = routing_key_predicate
        self.counter = AtomicInteger(0)

        # Initialize queue and running tasks
        self.queue = queue.PriorityQueue(maxsize=queue_size)
        self.running_tasks = []

        # Setup process communication
        self.processor = Processor(app_config, search_config, self._processor_worker)
        self._shutdown = False

        # Start the processing thread
        self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()

    @staticmethod
    def _processor_worker(conn: Any, app_config: ApplicationConfig, search_config: SearchConfig) -> None:
        """Worker function that runs in separate process"""
        processor = ServiceProcessor(app_config, search_config)

        try:
            while True:
                if conn.poll():
                    try:
                        routing_key, correlation_id, message = conn.recv()
                        if routing_key is None:  # Shutdown signal
                            break
                        logging.set_correlation_id(correlation_id)
                        response_body = processor.process(routing_key, message)
                        conn.send(response_body)
                    except Exception as exc:
                        logger.exception("Failed to process message in worker", exc_info=exc)
                        conn.send(None)
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
        except Exception as exc:
            logger.exception("Processor worker encountered error", exc_info=exc)
        finally:
            conn.close()

    def __send_to_processor(self) -> Optional[ProcessingItem]:
        try:
            processing_item: ProcessingItem = self.queue.get_nowait()
            logging.set_correlation_id(processing_item.log_correlation_id)
            if self.routing_key_predicate and self.routing_key_predicate(processing_item.routing_key):
                return processing_item

            log_incoming_message(processing_item.routing_key, processing_item.msg_correlation_id, processing_item.item)

            # Send to processor
            self.processor.parent_conn.send(
                (processing_item.routing_key, processing_item.log_correlation_id, processing_item.item)
            )

            processing_item.send_time = time.time()  # Record send time for tracking

            # Add to running tasks
            self.running_tasks.append(processing_item)
            return processing_item
        except queue.Empty:
            return None
        except Exception as exc:
            logger.exception("Failed to send message to processor", exc_info=exc)
            return None

    def __receive_results(self) -> None:
        if self.running_tasks:
            try:
                if self.processor.parent_conn.poll(timeout=0.1):
                    response_body = self.processor.parent_conn.recv()

                    if self.running_tasks:
                        completed_task = self.running_tasks.pop(0)  # FIFO for completed tasks

                        # Handle response similar to original handler
                        self._handle_response(completed_task, response_body)
            except Exception as exc:
                logger.exception("Failed to receive response from processor", exc_info=exc)

    def _process_queue(self):
        """Thread function to process queue and communicate with processor"""
        while not self._shutdown:
            try:
                # Send messages to processor (up to prefetch_size)
                while len(self.running_tasks) < self.prefetch_size:
                    if not self.__send_to_processor():
                        break

                self.__receive_results()

                # Small sleep to prevent busy waiting
                if not self.running_tasks and self.queue.empty():
                    time.sleep(0.01)

            except Exception as exc:
                logger.exception("Error in processing thread", exc_info=exc)
                time.sleep(0.1)

    def _handle_response(self, processing_item: ProcessingItem, response_body: Any):
        """Handle the response from processor similar to original AmqpRequestHandler"""
        if response_body is None:
            return None
        logging.set_correlation_id(processing_item.log_correlation_id)
        try:
            if processing_item.reply_to:
                log_outgoing_message(processing_item.reply_to, processing_item.msg_correlation_id, response_body)
                self.client.reply(processing_item.reply_to, processing_item.msg_correlation_id, response_body)
        except Exception as exc:
            logger.exception("Failed to publish result", exc_info=exc)
            return None

        logger.debug("Finished processing response")
        return None

    def handle_amqp_request(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        props: BasicProperties,
        body: bytes,
    ) -> None:
        """Function for handling amqp request: convert to ProcessingItem and add to priority queue."""
        number: int = self.counter.inc()

        # Processing request
        message = serialize_message(channel, method.delivery_tag, body)
        if not message:
            return None

        priority = get_priority(props)

        # Create ProcessingItem
        processing_item = ProcessingItem(
            priority=priority,
            number=number,
            routing_key=method.routing_key,
            reply_to=props.reply_to,
            log_correlation_id=logging.get_correlation_id(),
            msg_correlation_id=props.correlation_id,
            item=message,
        )

        # Add to priority queue (blocking on overflow)
        try:
            self.queue.put(processing_item, block=True)
        except Exception as exc:
            logger.exception("Failed to add item to processing queue", exc_info=exc)
            return None
        return None

    def shutdown(self):
        """Gracefully shutdown the handler"""
        self._shutdown = True

        # Stop processor process
        self.processor.shutdown()

        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)
