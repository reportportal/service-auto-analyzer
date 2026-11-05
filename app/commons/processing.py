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
"""Worker processes which run tasks for the coordinator, one task at a time each."""

import json
import multiprocessing
from multiprocessing.connection import Connection
from typing import Any, Callable, Optional

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem, ProcessingResult

LOGGER = logging.getLogger("analyzerApp.processing")

# The coordinator runs threads (the default executor of the event loop), and forking a multithreaded process is unsafe,
# so workers start from a fork server. Its targets must be picklable.
MP_CONTEXT = multiprocessing.get_context("forkserver")
TERMINATE_TIMEOUT = 5.0


class RealProcessor:
    """A worker process with a pipe to it."""

    parent_conn: Connection
    process: Any

    def __init__(self, target: Callable[[Connection], None]) -> None:
        """Start the worker process.

        :param target: Function to run in the process, it receives the child end of the pipe
        """
        self.parent_conn, child_conn = MP_CONTEXT.Pipe()
        self.process = MP_CONTEXT.Process(target=target, args=[child_conn], daemon=True)
        self.process.start()
        # Close the child end in this process, so the parent end reports EOF when the worker dies
        child_conn.close()

    @property
    def pid(self) -> Optional[int]:
        return self.process.pid

    def is_alive(self) -> bool:
        return self.process.is_alive()

    def fileno(self) -> int:
        """File descriptor of the parent end of the pipe, readable when a result or EOF is available."""
        return self.parent_conn.fileno()

    def send(self, item: ProcessingItem) -> None:
        self.parent_conn.send(item)

    def poll(self, timeout: float = 0.0) -> bool:
        return self.parent_conn.poll(timeout)

    def recv(self) -> Any:
        return self.parent_conn.recv()

    def kill(self) -> None:
        """Stop the process at once: SIGTERM, then SIGKILL if it doesn't exit in time."""
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=TERMINATE_TIMEOUT)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=TERMINATE_TIMEOUT)
        self._close()

    def shutdown(self) -> None:
        """Ask the process to exit after its current task, and kill it if it doesn't exit in time."""
        if self.process.is_alive():
            try:
                self.parent_conn.send(None)  # Shutdown signal
                self.process.join(timeout=TERMINATE_TIMEOUT)
            except Exception as exc:
                LOGGER.exception("Error shutting down processor process", exc_info=exc)
        self.kill()

    def _close(self) -> None:
        try:
            self.parent_conn.close()
        except OSError:
            pass


class Worker:
    """Function object which runs in a worker process and processes tasks through ``ServiceProcessor``."""

    app_config: ApplicationConfig
    search_config: SearchConfig
    _init_services: set[str]

    def __init__(
        self,
        app_config: ApplicationConfig,
        search_config: SearchConfig,
        init_services: set[str],
    ):
        """Initialize the worker.

        :param app_config: Application configuration
        :param search_config: Search configuration for the processor
        :param init_services: Routing keys of the services to initialize in the worker process
        """
        self.app_config = app_config
        self.search_config = search_config
        self._init_services = init_services

    @staticmethod
    def process(processing_item: ProcessingItem, processor: Any) -> ProcessingResult:
        """Process one task.

        :param processing_item: The task, with a raw JSON body
        :param processor: The ``ServiceProcessor`` to run it with
        :return: The result of the task
        """
        routing_key = processing_item.routing_key
        try:
            body = json.loads(processing_item.item, strict=False)
            result_value = processor.process(routing_key, body)
            return ProcessingResult(processing_item.number, result_value, success=True)
        except Exception as exc:
            LOGGER.exception(f"Failed to process message '{routing_key}' in worker", exc_info=exc)
            return ProcessingResult(processing_item.number, None, success=False, error=exc)

    def work(self, conn: Connection) -> None:
        """Process tasks from the pipe until the shutdown signal or until the pipe is closed.

        :param conn: Child end of the pipe to the coordinator
        """
        # A worker started from a fork server doesn't inherit the logging configuration
        logging.setup(self.app_config)
        # Imported here: the services pull in the ML libraries, which the coordinator process must not load
        from app.service.processor import ServiceProcessor

        processor = ServiceProcessor(self.app_config, self.search_config, services_to_init=self._init_services)
        try:
            while True:
                processing_item: Optional[ProcessingItem] = conn.recv()
                if not processing_item:  # Shutdown signal
                    break
                logging.set_correlation_id(processing_item.log_correlation_id)
                conn.send(self.process(processing_item, processor))
        except EOFError:
            LOGGER.info("Coordinator closed the pipe, worker exits")
        except Exception as exc:
            LOGGER.exception("Processor worker encountered error", exc_info=exc)
        conn.close()
