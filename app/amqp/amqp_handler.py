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
import threading
import time
from queue import Empty, PriorityQueue
from typing import Any, Callable, Optional, Protocol

from pika.adapters.blocking_connection import BlockingChannel
from pika.spec import Basic, BasicProperties

from app.amqp.amqp import AmqpClient
from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem, ProcessingResult
from app.commons.processing import DummyProcessor, Processor, RealProcessor
from app.service.processor import ServiceProcessor

logger = logging.getLogger("analyzerApp.amqpHandler")


def log_incoming_message(routing_key: str, correlation_id: str, body: Any) -> None:
    body_str = json.dumps(body)
    logger.debug(
        f"Processing message: --Routing key: {routing_key} --Correlation ID: {correlation_id} --Body: {body_str}"
    )


def log_outgoing_message(reply_to: str, correlation_id: str, body: Any) -> None:
    body_str = json.dumps(body)
    logger.debug(f"Replying message: --To: {reply_to} --Correlation ID: {correlation_id} --Body: {body_str}")


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


class AmqpRequestHandler(Protocol):
    processor: Processor
    running_tasks: list[ProcessingItem]

    def handle_amqp_request(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        props: BasicProperties,
        body: bytes,
    ) -> None:
        """Handle AMQP request."""
        ...


class ProcessAmqpRequestHandler:
    """Class for handling AMQP requests with process-based routing and priority queue."""

    app_config: ApplicationConfig
    search_config: SearchConfig
    client: AmqpClient
    queue_size: int
    routing_key_predicate: Callable[[str], bool]
    counter: AtomicInteger
    queue: PriorityQueue[ProcessingItem]
    running_tasks: list[ProcessingItem]
    processor: Processor
    _processing_thread: Optional[threading.Thread]
    _shutdown: bool
    _init_services: set[str]

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        *,
        queue_size: int = 100,
        prefetch_size: int = 2,
        routing_key_predicate: Optional[Callable[[str], bool]] = None,
        client: Optional[AmqpClient] = None,
        init_services: Optional[list[str]] = None,
        name: str = None,
    ):
        """Initialize handler for processing requests with process-based communication.

        :param app_config: Application configuration object
        :param search_config: Search configuration object
        :param queue_size: Maximum size of the internal priority queue (buffer)
        :param routing_key_predicate: Optional predicate to filter routing keys for processing. Should return True for
        keys to process.
        :param client: Optional AMQP client for sending replies (useful for testing)
        :param init_services: Optional list of routing keys to initialize the processor with specific routing keys
        """
        self.app_config = app_config
        self.search_config = search_config
        self.client = client or AmqpClient(app_config)
        self.queue_size = queue_size
        self.routing_key_predicate = routing_key_predicate or (lambda _: True)
        if init_services:
            self._init_services = set(init_services)
        else:
            self._init_services = set()
        self.counter = AtomicInteger(0)

        # Initialize queue and running tasks
        self.queue = PriorityQueue(maxsize=queue_size)
        self.running_tasks = []

        # Setup process communication
        self.processor = RealProcessor(
            app_config, search_config, self._processor_worker, init_services=self._init_services
        )
        self._shutdown = False

        # Start the processing thread
        self._processing_thread = threading.Thread(target=self._process_queue, name=name, daemon=True)
        self._processing_thread.start()

    @staticmethod
    def _processor_worker(
        conn: Any, app_config: ApplicationConfig, search_config: SearchConfig, init_services: Optional[set[str]] = None
    ) -> None:
        """Worker function that runs in separate process"""
        processor = ServiceProcessor(app_config, search_config, services_to_init=init_services)
        try:
            while True:
                if conn.poll():
                    processing_item: ProcessingItem = conn.recv()
                    if not processing_item:  # Shutdown signal
                        break
                    routing_key = processing_item.routing_key

                    logging.set_correlation_id(processing_item.log_correlation_id)

                    result = None
                    for i in range(app_config.amqpHandlerMaxRetries):
                        try:
                            result_value = processor.process(routing_key, processing_item.item)
                            result = ProcessingResult(processing_item, result_value, success=True, retry_count=i)
                            break
                        except Exception as exc:
                            logger.exception(
                                f"Failed to process message {routing_key}' in worker on attempt {i + 1} of "
                                + str(app_config.amqpHandlerMaxRetries),
                                exc_info=exc,
                            )
                            result = ProcessingResult(processing_item, None, success=False, error=exc, retry_count=i)
                            time.sleep(0.01)  # Small sleep to prevent busy waiting
                    conn.send(result)
                else:
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
        except Exception as exc:
            logger.exception("Processor worker encountered error", exc_info=exc)
        conn.close()

    def __send_task(self, processing_item: ProcessingItem) -> None:
        """Send processing item to processor process"""
        processing_item.send_time = time.time()  # Record send time for tracking
        self.processor.send(processing_item)
        self.running_tasks.append(processing_item)

    def __send_to_processor(self) -> Optional[ProcessingItem]:
        try:
            processing_item: ProcessingItem = self.queue.get_nowait()
            logging.set_correlation_id(processing_item.log_correlation_id)
            if not self.routing_key_predicate(processing_item.routing_key):
                return processing_item
            log_incoming_message(processing_item.routing_key, processing_item.msg_correlation_id, processing_item.item)
            self.__send_task(processing_item)
            return processing_item
        except Empty:
            return None
        except Exception as exc:
            logger.exception("Failed to send message to processor", exc_info=exc)
            return None

    def _restart_processor(self) -> None:
        """Restart the processor process if it has died"""
        # Store running tasks to re-send them
        tasks_to_resend = self.running_tasks.copy()

        # Increment retries for the first task, since it might be the one that caused the failure
        tasks_to_resend[0].retries += 1

        # Clean up old process and connections
        self.processor.shutdown()

        # Clear running tasks list before re-sending
        self.running_tasks.clear()

        # Create new processor instance
        self.processor = RealProcessor(
            self.app_config, self.search_config, self._processor_worker, self._init_services
        )
        logger.info("Successfully restarted processor process")

        # Re-send all previously running tasks to the new processor
        for task in tasks_to_resend:
            if task.retries >= self.app_config.amqpHandlerMaxRetries:
                logger.warning(f"Task {task.routing_key} - {task.msg_correlation_id} exceeded max retries, skipping")
                continue
            try:
                self.__send_task(task)
                logger.debug(f"Re-sent task to new processor: {task.routing_key} - {task.msg_correlation_id}")
            except Exception as exc:
                logger.exception(f"Failed to re-send task {task.routing_key} to new processor", exc_info=exc)
        logger.info(f"Re-sent {len(self.running_tasks)} tasks to new processor")

    def __has_long_running_tasks(self) -> bool:
        """Check if there are any long-running tasks that need attention"""
        if not self.running_tasks:
            return False

        # Check if any task has been running longer than the configured timeout
        for task in self.running_tasks:
            if task.send_time and (time.time() - task.send_time) > self.app_config.amqpHandlerTaskTimeout:
                logger.warning(f"Task {task.routing_key} - {task.msg_correlation_id} is taking too long")
                return True
        return False

    def __handle_response(self, result: ProcessingResult):
        response_body = result.result
        if response_body is None:
            return None
        logging.set_correlation_id(result.item.log_correlation_id)
        try:
            if result.item.reply_to:
                log_outgoing_message(result.item.reply_to, result.item.msg_correlation_id, response_body)
                self.client.reply(result.item.reply_to, result.item.msg_correlation_id, response_body)
        except Exception as exc:
            logger.exception("Failed to publish result", exc_info=exc)
            return None

        logger.debug("Finished processing response")
        return None

    def __receive_results(self) -> None:
        if self.running_tasks:
            try:
                if self.processor.poll(timeout=0.1):
                    result: ProcessingResult = self.processor.recv()
                    if self.running_tasks:
                        completed_task = self.running_tasks.pop(0)  # FIFO for completed tasks
                        if completed_task.msg_correlation_id != result.item.msg_correlation_id:
                            logger.warning(
                                f"Received result for task {result.item.msg_correlation_id} but expected "
                                f"{completed_task.msg_correlation_id}. Possible mismatch."
                            )
                    if result.retry_count > 0:
                        if result.success:
                            logger.warning(
                                f"Task {result.item.routing_key} - {result.item.msg_correlation_id} "
                                f"was retried {result.retry_count} times"
                            )
                        else:
                            logger.error(
                                f"Task {result.item.routing_key} - {result.item.msg_correlation_id} failed after "
                                f"{result.retry_count} retries."
                            )
                    self.__handle_response(result)
            except Exception as exc:
                logger.exception("Failed to receive response from processor", exc_info=exc)

    def _process_queue(self):
        """Thread function to process queue and communicate with processor"""
        while not self._shutdown:
            try:
                # Check if processor process is alive and restart if needed
                if not self.processor.is_alive():
                    logger.warning("Processor process is not running, restarting...")
                    self._restart_processor()

                if self.__has_long_running_tasks():
                    logger.warning("Detected long-running tasks, restarting processor...")
                    self._restart_processor()

                # Send messages to processor (up to prefetch_size)
                while not self.running_tasks:
                    if not self.__send_to_processor():
                        break

                self.__receive_results()

                # Small sleep to prevent busy waiting
                if not self.running_tasks and self.queue.empty():
                    time.sleep(0.01)

            except Exception as exc:
                logger.exception("Error in processing thread", exc_info=exc)
                time.sleep(0.1)

    def handle_amqp_request(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        props: BasicProperties,
        body: bytes,
    ) -> None:
        """Function for handling amqp request: convert to ProcessingItem and add to priority queue."""
        number: int = self.counter.inc()
        logging.new_correlation_id()

        # Processing request
        message = serialize_message(channel, method.delivery_tag, body)
        if not message:
            return None

        priority = get_priority(props)

        # Create ProcessingItem
        processing_item = ProcessingItem(
            priority=priority,
            number=number,
            routing_key=method.routing_key or "",
            reply_to=props.reply_to,
            log_correlation_id=logging.get_correlation_id(),
            msg_correlation_id=props.correlation_id or "",
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


class DirectAmqpRequestHandler:
    """Class for direct handling AMQP requests. Use for debug purpose."""

    app_config: ApplicationConfig
    search_config: SearchConfig
    routing_key_predicate: Callable[[str], bool]
    running_tasks: list[ProcessingItem]
    counter: int
    processor: Processor
    _init_services: set[str]
    _processor: ServiceProcessor

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        *,
        routing_key_predicate: Optional[Callable[[str], bool]] = None,
        init_services: Optional[list[str]] = None,
    ):
        """Initialize handler for processing requests.

        :param app_config: Application configuration object
        :param search_config: Search configuration object
        :param routing_key_predicate: Optional predicate to filter routing keys for processing. Should return True for
        keys to process.
        :param init_services: Optional list of routing keys to initialize the processor with specific routing keys
        """
        self.app_config = app_config
        self.search_config = search_config
        self.routing_key_predicate = routing_key_predicate or (lambda _: True)
        self.running_tasks = []
        if init_services:
            self._init_services = set(init_services)
        else:
            self._init_services = set()
        self.counter = 0
        self.processor = DummyProcessor()
        self._processor = ServiceProcessor(app_config, search_config, services_to_init=self._init_services)

    def handle_amqp_request(
        self,
        channel: BlockingChannel,
        method: Basic.Deliver,
        props: BasicProperties,
        body: bytes,
    ) -> None:
        """Function for handling amqp request: index, search and analyze using routing key."""
        logging.new_correlation_id()

        # Processing request
        message = serialize_message(channel, method.delivery_tag, body)
        if not message:
            return None

        routing_key = method.routing_key or ""
        if not self.routing_key_predicate(routing_key):
            return None

        self.counter += 1
        log_incoming_message(routing_key, props.correlation_id or "", message)

        self.running_tasks.append(
            ProcessingItem(
                priority=get_priority(props),
                number=self.counter,
                routing_key=routing_key,
                reply_to=props.reply_to,
                log_correlation_id=logging.get_correlation_id(),
                msg_correlation_id=props.correlation_id or "",
                item=message,
            )
        )

        # Process using the processor
        try:
            response_body = self._processor.process(method.routing_key or "", message)
        except Exception as exc:
            logger.exception("Failed to process message", exc_info=exc)
            self.running_tasks.pop(0)
            return None

        # Sending response if applicable
        if response_body is None:
            self.running_tasks.pop(0)
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
            self.running_tasks.pop(0)
            return None
        logger.debug("Finished processing response")
        self.running_tasks.pop(0)
        return None
