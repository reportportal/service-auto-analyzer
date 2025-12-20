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

from time import time
from typing import Optional

from app.commons import logging, request_factory
from app.commons.model.launch_objects import ApplicationConfig, Launch
from app.commons.namespace_finder import NamespaceFinder
from app.utils import utils

LOGGER = logging.getLogger("analyzerApp.namespaceFinderService")


class NamespaceFinderService:
    namespace_finder: NamespaceFinder

    def __init__(
        self,
        app_config: ApplicationConfig,
        *,
        namespace_finder: Optional[NamespaceFinder] = None,
    ):
        self.namespace_finder = namespace_finder or NamespaceFinder(app_config)

    @utils.ignore_warnings
    def update_chosen_namespaces(self, launches: list[Launch]) -> None:
        project_ids_str = ", ".join({str(launch.project) for launch in launches})
        launch_ids_str = ", ".join({str(launch.launchId) for launch in launches})
        LOGGER.info(f"Started updating chosen namespaces for projects '{project_ids_str}', launches: {launch_ids_str}")
        t_start = time()
        log_words, project_id = request_factory.prepare_log_words(launches)
        LOGGER.debug(f"Project id {project_id}")
        if project_id is not None:
            self.namespace_finder.update_namespaces(project_id, log_words)
        LOGGER.info("Finished updating chosen namespaces %.2f s", time() - t_start)
