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
from queue import Empty, PriorityQueue, Queue
from typing import Any, Callable, Optional, Protocol

from elasticsearch import ConflictError
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


def retry(item: ProcessingItem, exc: Exception) -> bool:
    """Default retry predicate for handling exceptions."""
    # You can customize this logic based on your requirements
    if isinstance(exc, ConflictError) and item.routing_key == "item_remove":
        return "but no document was found" not in exc.error
    return True


class Worker:
    """Worker class for processing AMQP messages in a separate process.

    Handles initialization of services and provides retry logic for failed processing attempts.
    Contains the main worker function that runs in a separate process and processes messages.
    """

    _init_services: set[str]
    _retry_predicate: Optional[Callable[[ProcessingItem, Exception], bool]]

    def __init__(
        self,
        init_services: set[str],
        retry_predicate: Optional[Callable[[ProcessingItem, Exception], bool]] = None,
    ):
        """Initialize worker with services to initialize and optional retry predicate.

        :param set[str] init_services: Set of service names to initialize in the worker process
        :param Optional[Callable[[ProcessingItem, Exception], bool]] retry_predicate: Optional function to determine
        if a failed processing attempt should be retried
        """
        self._init_services = init_services
        self._retry_predicate = retry_predicate

    def __process(
        self, processing_item: ProcessingItem, processor: ServiceProcessor, max_retries: int
    ) -> Optional[ProcessingResult]:
        result = None
        routing_key = processing_item.routing_key
        for i in range(max_retries):
            try:
                result_value = processor.process(routing_key, processing_item.item)
                result = ProcessingResult(processing_item, result_value, success=True, retry_count=i)
                break
            except Exception as exc:
                logger.exception(
                    f"Failed to process message '{routing_key}' in worker on attempt {i + 1} of {max_retries}",
                    exc_info=exc,
                )
                result = ProcessingResult(processing_item, None, success=False, error=exc, retry_count=i)
                if self._retry_predicate is not None:
                    if not self._retry_predicate(processing_item, exc):
                        logger.info(
                            f"Retry predicate failed for message {routing_key} - "
                            f"{processing_item.msg_correlation_id}, not retrying"
                        )
                        break
                time.sleep(0.01)  # Small sleep to prevent busy waiting
        return result

    def work(
        self,
        conn: Any,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
    ) -> None:
        """Worker function that runs in separate process.

        Continuously polls for processing items from the connection, processes them using ServiceProcessor,
        handles retries based on the retry predicate, and sends results back through the connection.

        :param Any conn: Process connection object for communication with parent process
        :param ApplicationConfig app_config: Application configuration containing retry settings
        :param SearchConfig search_config: Search configuration for the processor
        """
        processor = ServiceProcessor(app_config, search_config, services_to_init=self._init_services)
        try:
            while True:
                if not conn.poll():
                    # If no message is available, wait for a short time before checking again
                    time.sleep(0.01)  # Small sleep to prevent busy waiting
                    continue

                processing_item: ProcessingItem = conn.recv()
                if not processing_item:  # Shutdown signal
                    break

                logging.set_correlation_id(processing_item.log_correlation_id)

                result = self.__process(processing_item, processor, app_config.amqpHandlerMaxRetries)
                conn.send(result)
        except Exception as exc:
            logger.exception("Processor worker encountered error", exc_info=exc)
        conn.close()


class AtomicInteger:
    """Thread-safe integer counter with atomic increment and decrement operations.

    Provides thread-safe operations for incrementing, decrementing, and reading integer values
    using a threading lock to ensure atomicity across multiple threads.
    """

    def __init__(self, value: int = 0) -> None:
        """Initialize the atomic integer with an optional starting value.

        :param int value: Initial value for the counter (default: 0)
        """
        self._value = int(value)
        self._lock = threading.Lock()

    def inc(self, d: int = 1) -> int:
        """Atomically increment the value and return the new value.

        :param int d: Amount to increment by (default: 1)
        :return: The new value after incrementing
        :rtype: int
        """
        with self._lock:
            self._value += int(d)
            return self._value

    def dec(self, d: int = 1) -> int:
        """Atomically decrement the value and return the new value.

        :param int d: Amount to decrement by (default: 1)
        :return: The new value after decrementing
        :rtype: int
        """
        return self.inc(-d)

    @property
    def value(self) -> int:
        """Get the current value in a thread-safe manner.

        :return: Current integer value
        :rtype: int
        """
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
    __running_tasks: Queue[ProcessingItem]
    processor: Processor
    _monitor_thread: Optional[threading.Thread]
    _sender_thread: Optional[threading.Thread]
    _receiver_thread: Optional[threading.Thread]
    _shutdown: bool
    _init_services: set[str]
    _retry_predicate: Optional[Callable[[ProcessingItem, Exception], bool]]

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        *,
        queue_size: int = 100,
        routing_key_predicate: Optional[Callable[[str], bool]] = None,
        client: Optional[AmqpClient] = None,
        init_services: Optional[list[str]] = None,
        retry_predicate: Optional[Callable[[ProcessingItem, Exception], bool]] = retry,
        name: Optional[str] = None,
    ):
        """Initialize handler for processing requests with process-based communication.

        :param ApplicationConfig app_config: Application configuration object
        :param SearchConfig search_config: Search configuration object
        :param int queue_size: Maximum size of the internal priority queue (buffer)
        :param Optional[Callable[[str], bool]] routing_key_predicate: Optional predicate to filter routing keys for
        processing. Should return True for keys to process.
        :param Optional[AmqpClient] client: Optional AMQP client for sending replies (useful for testing)
        :param Optional[list[str]] init_services: Optional list of routing keys to initialize the processor with
        specific routing keys
        :param Optional[Callable[[ProcessingItem, Exception], bool]] retry_predicate: Optional function to determine
        if a failed processing attempt should be retried
        :param Optional[str] name: Optional name for the processing thread
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
        self._retry_predicate = retry_predicate
        self.counter = AtomicInteger(0)

        # Initialize queue and running tasks
        self.queue = PriorityQueue(maxsize=queue_size)
        self.__running_tasks = Queue()

        # Setup process communication
        self.processor = RealProcessor(
            app_config,
            search_config,
            Worker(self._init_services, self._retry_predicate).work,
        )
        self._shutdown = False

        # Start the three processing threads
        thread_name_prefix = name or "ProcessHandler"
        self._monitor_thread = threading.Thread(
            target=self._monitor_processor, name=f"{thread_name_prefix}-Monitor", daemon=True
        )
        self._sender_thread = threading.Thread(
            target=self._send_tasks, name=f"{thread_name_prefix}-Sender", daemon=True
        )
        self._receiver_thread = threading.Thread(
            target=self._receive_results, name=f"{thread_name_prefix}-Receiver", daemon=True
        )

        self._monitor_thread.start()
        self._sender_thread.start()
        self._receiver_thread.start()

    def __send_task(self, processing_item: ProcessingItem) -> None:
        """Send processing item to processor process"""
        processing_item.send_time = time.time()  # Record send time for tracking
        self.processor.send(processing_item)
        self.__running_tasks.put(processing_item)

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
        if self._shutdown:
            # If shutdown is initiated, do not restart the processor
            return

        # Store running tasks to re-send them, clearing the queue
        if not self.__running_tasks.queue:
            tasks_to_resend = []
        else:
            with self.__running_tasks.mutex:
                if not self.__running_tasks.queue:
                    tasks_to_resend = []
                else:
                    tasks_to_resend = list(self.__running_tasks.queue)
                    self.__running_tasks.queue.clear()

        if tasks_to_resend:
            # Increment retries for the first task, since it might be the one that caused the failure
            tasks_to_resend[0].retries += 1

        # Clean up old process and connections
        self.processor.shutdown()

        # Create new processor instance
        self.processor = RealProcessor(
            self.app_config,
            self.search_config,
            Worker(self._init_services, self._retry_predicate).work,
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
        logger.info(f"Re-sent {self.__running_tasks.qsize()} tasks to new processor")

    def __has_long_running_tasks(self) -> bool:
        """Check if there are any long-running tasks that need attention"""
        if not self.__running_tasks.queue:
            return False

        # Check if any task has been running longer than the configured timeout
        with self.__running_tasks.mutex:
            if not self.__running_tasks.queue:
                return False
            for task in self.__running_tasks.queue:
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

    def __process_result(self, result: ProcessingResult) -> None:
        if not self.__running_tasks.empty():
            try:
                completed_task = self.__running_tasks.get()  # FIFO for completed tasks
                if completed_task.msg_correlation_id != result.item.msg_correlation_id:
                    logger.warning(
                        f"Received result for task {result.item.msg_correlation_id} but expected "
                        f"{completed_task.msg_correlation_id}. Possible mismatch."
                    )
            except Empty:
                logger.warning("Received result but no tasks in running queue, possible bug")
        if result.success and result.retry_count > 0:
            logger.warning(
                f"Task {result.item.routing_key} - {result.item.msg_correlation_id} "
                f"was retried {result.retry_count} times"
            )
        elif not result.success:
            logger.error(
                f"Task {result.item.routing_key} - {result.item.msg_correlation_id} failed after "
                f"{result.retry_count} retries."
            )
        self.__handle_response(result)

    def __receive_results(self) -> None:
        try:
            if not self.processor.poll(timeout=0.1):
                return
            result: ProcessingResult = self.processor.recv()
            self.__process_result(result)
        except Exception as exc:
            logger.exception("Failed to receive response from processor", exc_info=exc)

    def _monitor_processor(self):
        """Thread 1: Monitor processor and task states"""
        while not self._shutdown:
            try:
                # Check if processor process is alive and restart if needed
                if not self.processor.is_alive():
                    logger.warning("Processor process is not running, restarting...")
                    self._restart_processor()

                if self.__has_long_running_tasks():
                    logger.warning("Detected long-running tasks, restarting processor...")
                    self._restart_processor()

                # Monitor interval
                time.sleep(0.1)

            except Exception as exc:
                logger.exception("Error in monitor thread", exc_info=exc)
                time.sleep(0.1)

    def _send_tasks(self):
        """Thread 2: Send tasks to processor if running tasks < 2"""
        while not self._shutdown:
            try:
                # Send messages to processor if we have less than 2 running tasks
                while self.__running_tasks.qsize() < 2:
                    if not self.__send_to_processor():
                        break

                # Small sleep to prevent busy waiting
                if self.queue.empty():
                    time.sleep(0.01)

            except Exception as exc:
                logger.exception("Error in sender thread", exc_info=exc)
                time.sleep(0.1)

    def _receive_results(self):
        """Thread 3: Receive results from processor if there are running tasks"""
        while not self._shutdown:
            try:
                if not self.__running_tasks.empty():
                    self.__receive_results()
                else:
                    # Small sleep when no running tasks
                    time.sleep(0.01)

            except Exception as exc:
                logger.exception("Error in receiver thread", exc_info=exc)
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

    @property
    def running_tasks(self) -> list[ProcessingItem]:
        """Get the queue of currently running tasks."""
        if not self.__running_tasks.queue:
            return []
        with self.__running_tasks.mutex:
            if not self.__running_tasks.queue:
                return []
            return list(self.__running_tasks.queue)

    def shutdown(self):
        """Gracefully shutdown the handler"""
        self._shutdown = True

        # Stop processor process
        self.processor.shutdown()

        # Wait for all processing threads
        for thread in [self._monitor_thread, self._sender_thread, self._receiver_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5)


class DirectAmqpRequestHandler:
    """Class for direct handling AMQP requests. Use for debug purpose."""

    app_config: ApplicationConfig
    search_config: SearchConfig
    routing_key_predicate: Callable[[str], bool]
    __running_tasks: Queue[ProcessingItem]
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
        self.__running_tasks = Queue()
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

        self.__running_tasks.put(
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
            self.__running_tasks.get()
            return None

        # Sending response if applicable
        if response_body is None:
            self.__running_tasks.get()
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
            self.__running_tasks.get()
            return None
        logger.debug("Finished processing response")
        self.__running_tasks.get()
        return None

    @property
    def running_tasks(self) -> list[ProcessingItem]:
        """Get the queue of currently running tasks."""
        if not self.__running_tasks.queue:
            return []
        with self.__running_tasks.mutex:
            if not self.__running_tasks.queue:
                return []
            return list(self.__running_tasks.queue)
