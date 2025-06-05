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
from multiprocessing import Pipe, Process
from typing import Any, Optional

from amqp.amqp import AmqpClient
from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem
from app.service.processor import Processor

logger = logging.getLogger("analyzerApp.amqpHandler")


def log_incoming_message(method: Basic.Deliver, props: BasicProperties, body: bytes) -> None:
    logging.new_correlation_id()
    logger.debug(f"Processing message: --Method: {method} --Properties: {props} --Body: {body}")


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


class AmqpRequestHandler:
    """Class for handling AMQP requests with service routing based on routing key"""

    app_config: ApplicationConfig
    search_config: SearchConfig
    _processor: Processor

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
        log_incoming_message(method, props, body)

        # Processing request
        message = serialize_message(channel, method.delivery_tag, body)
        if not message:
            return None

        # Process using the processor
        try:
            response_body = self._processor.process(method.routing_key, message)
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


class ProcessAmqpRequestHandler:
    """Class for handling AMQP requests with process-based routing and priority queue"""

    app_config: ApplicationConfig
    search_config: SearchConfig
    client: AmqpClient
    queue_size: int
    prefetch_size: int
    counter: AtomicInteger
    queue: queue.PriorityQueue
    running_tasks: list
    parent_conn: Any
    child_conn: Any
    _processor_process: Optional[Process]
    _processing_thread: Optional[threading.Thread]
    _shutdown: bool

    def __init__(
        self, app_config: ApplicationConfig, search_config: SearchConfig, queue_size: int = 100, prefetch_size: int = 2
    ):
        """Initialize processor for handling requests with process-based communication"""
        self.app_config = app_config
        self.search_config = search_config
        self.client = AmqpClient(app_config)
        self.queue_size = queue_size
        self.prefetch_size = prefetch_size
        self.counter = AtomicInteger(0)

        # Initialize queue and running tasks
        self.queue: queue.PriorityQueue = queue.PriorityQueue(maxsize=queue_size)
        self.running_tasks: list = []

        # Setup process communication
        self.parent_conn, self.child_conn = Pipe()
        self._processor_process = None
        self._processing_thread = None
        self._shutdown = False

        # Start the processor process
        self._start_processor_process()

        # Start the processing thread
        self._processing_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._processing_thread.start()

    def _start_processor_process(self):
        """Start the processor in a separate process"""
        self._processor_process = Process(
            target=self._processor_worker, args=(self.child_conn, self.app_config, self.search_config)
        )
        self._processor_process.start()

    @staticmethod
    def _processor_worker(conn, app_config: ApplicationConfig, search_config: SearchConfig):
        """Worker function that runs in separate process"""
        processor = Processor(app_config, search_config)

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

    def _process_queue(self):
        """Thread function to process queue and communicate with processor"""
        while not self._shutdown:
            try:
                # Send messages to processor (up to prefetch_size)
                sent_count = 0
                while sent_count < self.prefetch_size and len(self.running_tasks) < self.prefetch_size:
                    try:
                        processing_item = self.queue.get_nowait()

                        # Send to processor
                        self.parent_conn.send(
                            (processing_item.routing_key, processing_item.log_correlation_id, processing_item.item)
                        )

                        # Add to running tasks
                        self.running_tasks.append(processing_item)
                        sent_count += 1

                    except queue.Empty:
                        break
                    except Exception as exc:
                        logger.exception("Failed to send message to processor", exc_info=exc)
                        break

                # Receive results from processor
                if self.running_tasks:
                    try:
                        if self.parent_conn.poll(timeout=0.1):
                            response_body = self.parent_conn.recv()

                            if self.running_tasks:
                                completed_task = self.running_tasks.pop(0)  # FIFO for completed tasks

                                # Handle response similar to original handler
                                self._handle_response(completed_task, response_body)
                    except Exception as exc:
                        logger.exception("Failed to receive response from processor", exc_info=exc)

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

        log_incoming_message(method, props, body)

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
        if self._processor_process and self._processor_process.is_alive():
            try:
                self.parent_conn.send((None, None))  # Shutdown signal
                self._processor_process.join(timeout=5)
                if self._processor_process.is_alive():
                    self._processor_process.terminate()
            except Exception as exc:
                logger.exception("Error shutting down processor process", exc_info=exc)

        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=5)

        # Close connections
        try:
            self.parent_conn.close()
        except Exception:
            pass
