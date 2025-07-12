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
from multiprocessing import Pipe, Process
from typing import Any, Callable, Protocol

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model.processing import ProcessingItem

logger = logging.getLogger("analyzerApp.amqpHandler")


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
                logger.exception("Error shutting down processor process", exc_info=exc)

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
