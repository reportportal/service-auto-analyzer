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

from app.commons import logging, namespace_finder
from app.commons.log_preparation import LogPreparation
from app.utils import utils

logger = logging.getLogger("analyzerApp.namespaceFinderService")


class NamespaceFinderService:

    def __init__(self, app_config=None, search_cfg=None):
        self.app_config = app_config or {}
        self.search_cfg = search_cfg or {}
        self.namespace_finder = namespace_finder.NamespaceFinder(self.app_config)
        self.log_preparation = LogPreparation()

    @utils.ignore_warnings
    def update_chosen_namespaces(self, launches):
        logger.info("Started updating chosen namespaces")
        t_start = time()
        log_words, project_id = self.log_preparation.prepare_log_words(launches)
        logger.debug("Project id %s", project_id)
        if project_id is not None:
            self.namespace_finder.update_namespaces(
                project_id, log_words)
        logger.info("Finished updating chosen namespaces %.2f s", time() - t_start)
