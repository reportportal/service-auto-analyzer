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

from multiprocessing import Pipe, Process
from typing import Any, Callable

from app.commons import logging
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig

logger = logging.getLogger("analyzerApp.amqpHandler")


class Processor:
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

    def shutdown(self) -> None:
        if self.process and self.process.is_alive():
            try:
                self.parent_conn.send((None, None))  # Shutdown signal
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
