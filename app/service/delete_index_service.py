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

from app.commons import logging
from app.commons.esclient import EsClient
from app.commons.model.launch_objects import ApplicationConfig, SearchConfig
from app.commons.model_chooser import ModelChooser
from app.commons.namespace_finder import NamespaceFinder
from app.commons.trigger_manager import TriggerManager
from app.utils import text_processing

LOGGER = logging.getLogger("analyzerApp.deleteIndexService")


class DeleteIndexService:
    app_config: ApplicationConfig
    search_cfg: SearchConfig
    namespace_finder: NamespaceFinder
    trigger_manager: TriggerManager
    es_client: EsClient
    model_chooser: ModelChooser

    def __init__(
        self,
        model_chooser: ModelChooser,
        app_config: ApplicationConfig,
        search_cfg: SearchConfig,
        *,
        es_client: Optional[EsClient] = None,
        namespace_finder: Optional[NamespaceFinder] = None,
        trigger_manager: Optional[TriggerManager] = None,
    ):
        """Initialize DeleteIndexService

        :param model_chooser: Model chooser instance
        :param app_config: Application configuration object
        :param search_cfg: Search configuration object
        :param es_client: Optional EsClient instance. If not provided, a new one will be created.
        """
        self.model_chooser = model_chooser
        self.app_config = app_config
        self.search_cfg = search_cfg
        self.es_client = es_client or EsClient(app_config=self.app_config)
        self.namespace_finder = namespace_finder or NamespaceFinder(self.app_config)
        self.trigger_manager = trigger_manager or TriggerManager(
            model_chooser, app_config=self.app_config, search_cfg=self.search_cfg
        )

    def delete_index(self, index_name: int) -> int:
        LOGGER.info("Started deleting index")
        t_start = time()
        is_index_deleted = self.es_client.delete_index(
            text_processing.unite_project_name(index_name, self.app_config.esProjectIndexPrefix)
        )
        self.namespace_finder.remove_namespaces(index_name)
        self.trigger_manager.delete_triggers(index_name)
        self.model_chooser.delete_all_custom_models(index_name)
        LOGGER.info("Finished deleting index %.2f s", time() - t_start)
        return int(is_index_deleted)
