#  Copyright 2025 EPAM Systems
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
import os
import time
from multiprocessing import Pipe, Process
from typing import Any, Callable, Optional, Protocol

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem, ProcessingResult
from app.service import ServiceProcessor

LOGGER = logging.getLogger("analyzerApp.processing")


class Processor(Protocol):
    @property
    def pid(self) -> int:
        """Return the process ID of the processor."""
        ...

    def is_alive(self) -> bool:
        """Check if the processor is still running."""
        ...

    def shutdown(self) -> None:
        """Shutdown the processor gracefully."""
        ...

    def send(self, item: ProcessingItem) -> None:
        """Send an item to the processor for processing."""
        ...

    def poll(self, timeout: float) -> bool:
        """Check if the processor has any response available within the timeout."""
        ...

    def recv(self) -> Any:
        """Receive a response from the processor."""
        ...


class DummyProcessor:
    _is_alive: bool

    def __init__(self) -> None:
        self._is_alive = True

    @property
    def pid(self) -> int:
        return os.getpid()

    def is_alive(self) -> bool:
        return self._is_alive

    def shutdown(self) -> None:
        self._is_alive = False

    def send(self, item: ProcessingItem) -> None:
        """Dummy send method that does nothing."""
        pass

    def poll(self, _: float) -> bool:
        return False

    def recv(self) -> Any:
        return object()


class RealProcessor:
    app_config: ApplicationConfig
    search_config: SearchConfig
    parent_conn: Any
    child_conn: Any
    process: Process

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        target: Callable[[Any, ApplicationConfig, SearchConfig], None],
    ) -> None:
        self.app_config = app_config
        self.search_config = search_config
        self.parent_conn, self.child_conn = Pipe()
        self.process = Process(target=target, args=(self.child_conn, self.app_config, self.search_config), daemon=True)
        self.process.start()

    @property
    def pid(self) -> int:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def shutdown(self) -> None:
        if self.process and self.process.is_alive():
            try:
                self.parent_conn.send(None)  # Shutdown signal
                self.process.join(timeout=5)
                if self.process.is_alive():
                    self.process.terminate()
            except Exception as exc:
                LOGGER.exception("Error shutting down processor process", exc_info=exc)

        # Close connections
        try:
            self.parent_conn.close()
        except Exception:
            pass

    def send(self, item: ProcessingItem) -> None:
        self.parent_conn.send(item)

    def poll(self, timeout: float) -> bool:
        return self.parent_conn.poll(timeout=timeout)

    def recv(self) -> Any:
        return self.parent_conn.recv()


class Worker:
    """Worker class for processing AMQP messages in a separate process.

    Handles initialization of services and provides retry logic for failed processing attempts.
    Contains the main worker function that runs in a separate process and processes messages.
    """

    _init_services: set[str]

    def __init__(
        self,
        init_services: set[str],
    ):
        """Initialize worker with services to initialize and optional retry predicate.

        :param set[str] init_services: Set of service names to initialize in the worker process
        if a failed processing attempt should be retried
        """
        self._init_services = init_services

    def __process(self, processing_item: ProcessingItem, processor: ServiceProcessor) -> Optional[ProcessingResult]:
        routing_key = processing_item.routing_key
        try:
            result_value = processor.process(routing_key, processing_item.item)
            return ProcessingResult(processing_item, result_value, success=True)
        except Exception as exc:
            LOGGER.exception(
                f"Failed to process message '{routing_key}' in worker",
                exc_info=exc,
            )
            return ProcessingResult(processing_item, None, success=False, error=exc)

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

                result = self.__process(processing_item, processor)
                conn.send(result)
        except Exception as exc:
            LOGGER.exception("Processor worker encountered error", exc_info=exc)
        conn.close()
